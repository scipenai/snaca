//! SSE → canonical `StreamEvent` translator for the OpenAI Responses API.
//!
//! ## Wire format
//!
//! Unlike Chat Completions, Responses SSE events carry an `event:` name:
//!
//! ```text
//! event: response.created
//! data: {"response":{"id":"resp_1","model":"gpt-5-mini"}}
//!
//! event: response.output_text.delta
//! data: {"output_index":0,"item_id":"msg_1","content_index":0,"delta":"Hi"}
//!
//! event: response.output_item.added
//! data: {"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"Read","arguments":""}}
//!
//! event: response.function_call_arguments.delta
//! data: {"output_index":1,"item_id":"fc_1","delta":"{\"path\":"}
//!
//! event: response.completed
//! data: {"response":{"id":"resp_1","status":"completed","usage":{...}}}
//! ```
//!
//! ## Synthesis
//!
//! Blocks are keyed by the wire's `output_index`, so interleaved text /
//! reasoning / tool-call items map to stable canonical block indices.
//! Text and thinking blocks open lazily on their first (non-empty)
//! delta — a message item that ends up with no text never produces an
//! empty canonical block. Tool-call blocks open eagerly on
//! `response.output_item.added` (that's where `call_id` / `name` live).
//!
//! `response.output_item.done` completes two things deltas can't: a
//! reasoning item's `encrypted_content` (attached as a `Signature`, opening
//! the Thinking block if no summary streamed), and a function call's full
//! `arguments` when an endpoint sent no argument deltas.
//!
//! Frames are dispatched on blank lines; a missing `event:` line falls
//! back to the payload's `type` field (some proxies strip event
//! names), and a trailing unterminated frame still dispatches at EOF.

use crate::error::{LlmError, LlmResult};
use crate::openai_responses::wire::{IncompleteDetails, OutputItem, WireUsage};
use crate::stream::{ContentBlockStart, ContentDelta, StreamEvent};
use bytes::Bytes;
use futures::stream::{BoxStream, Stream, StreamExt};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------- wire events ----------------

#[derive(Debug, Clone, Deserialize)]
pub struct ResponseEnvelope {
    #[serde(default)]
    pub response: Option<ResponseRef>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponseRef {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
    #[serde(default)]
    pub error: Option<serde_json::Value>,
}

/// Payload of `response.output_item.added` and `response.output_item.done`.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputItemEvent {
    #[serde(default)]
    pub output_index: Option<u32>,
    #[serde(default)]
    pub item: Option<OutputItem>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeltaEvent {
    #[serde(default)]
    pub output_index: Option<u32>,
    #[serde(default)]
    pub delta: Option<String>,
    /// Which reasoning-summary part a summary delta belongs to.
    #[serde(default)]
    pub summary_index: Option<u32>,
}

/// The `error` event: fields at the top level as documented, or nested
/// under `error` as some deployments send them.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorEvent {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<ErrorDetail>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorDetail {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
}

impl ErrorEvent {
    /// `code: message`, from whichever shape carried them.
    fn describe(self) -> String {
        let (nested_code, nested_message) =
            self.error.map(|e| (e.code, e.message)).unwrap_or_default();
        let message = self
            .message
            .or(nested_message)
            .unwrap_or_else(|| "stream error".into());
        match self.code.or(nested_code) {
            Some(code) => format!("{code}: {message}"),
            None => message,
        }
    }
}

// ---------------- state machine ----------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
}

/// What to open when a wire `output_index` is seen for the first time.
/// Tool blocks carry their `call_id` / `name` inline, so a tool block can
/// never be opened without them: an empty id would break the engine's
/// tool_use/tool_result pairing invariant, misroute the eager prerun's
/// result, and get the replayed history rejected by the provider.
enum BlockOpen {
    Text,
    Thinking,
    ToolUse { id: String, name: String },
}

impl BlockOpen {
    fn kind(&self) -> BlockKind {
        match self {
            BlockOpen::Text => BlockKind::Text,
            BlockOpen::Thinking => BlockKind::Thinking,
            BlockOpen::ToolUse { .. } => BlockKind::ToolUse,
        }
    }
}

#[derive(Default)]
pub struct StreamState {
    started: bool,
    finished: bool,
    next_index: u32,
    /// Wire `output_index` → (canonical block index, kind).
    blocks: HashMap<u32, (u32, BlockKind)>,
    /// Canonical indices of blocks opened but not yet closed, in
    /// creation order — closed all at once by the terminal events.
    open: BTreeSet<u32>,
    /// Wire `output_index`es that have streamed a refusal, so only the
    /// first refusal delta per item gets the `[refused] ` prefix.
    refusing: HashSet<u32>,
    /// Canonical tool blocks that received argument deltas; the rest take
    /// their arguments from the finished `output_item.done` item.
    args_streamed: HashSet<u32>,
    /// Wire `output_index` → last reasoning-summary part seen, so parts are
    /// separated by a blank line like the non-streaming converter does.
    summary_part: HashMap<u32, u32>,
}

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    fn next_idx(&mut self) -> u32 {
        let i = self.next_index;
        self.next_index += 1;
        i
    }

    /// Map (or create) the canonical block for a wire output_index.
    fn block_for(&mut self, output_index: u32, open: BlockOpen, out: &mut Vec<StreamEvent>) -> u32 {
        if let Some(&(canonical, _)) = self.blocks.get(&output_index) {
            return canonical;
        }
        let idx = self.next_idx();
        self.blocks.insert(output_index, (idx, open.kind()));
        self.open.insert(idx);
        let block = match open {
            BlockOpen::Text => ContentBlockStart::Text,
            BlockOpen::Thinking => ContentBlockStart::Thinking,
            BlockOpen::ToolUse { id, name } => ContentBlockStart::ToolUse { id, name },
        };
        out.push(StreamEvent::ContentBlockStart { index: idx, block });
        idx
    }

    /// Canonical index of a block already opened for this wire
    /// output_index, without creating one.
    fn existing_block(&self, output_index: u32) -> Option<u32> {
        self.blocks.get(&output_index).map(|&(idx, _)| idx)
    }

    /// Terminal failure: surface it as a stream error (the accumulator
    /// turns this into `LlmError::Provider`) and ignore every later
    /// frame. Used for provider-reported errors and for protocol
    /// violations we can't represent canonically.
    fn fail(&mut self, message: impl Into<String>, out: &mut Vec<StreamEvent>) {
        out.push(StreamEvent::Error {
            message: message.into(),
        });
        self.finished = true;
    }

    fn close_open_blocks(&mut self, out: &mut Vec<StreamEvent>) {
        for idx in std::mem::take(&mut self.open) {
            out.push(StreamEvent::ContentBlockStop { index: idx });
        }
    }

    fn finish(
        &mut self,
        status: Option<&str>,
        incomplete_details: Option<&IncompleteDetails>,
        usage: Option<WireUsage>,
        out: &mut Vec<StreamEvent>,
    ) {
        self.close_open_blocks(out);
        let has_tool_call = self
            .blocks
            .values()
            .any(|(_, kind)| *kind == BlockKind::ToolUse);
        let stop_reason = super::convert::stop_reason_for(
            status,
            incomplete_details.and_then(|d| d.reason.clone()),
            has_tool_call,
        );
        out.push(StreamEvent::MessageDelta {
            stop_reason: Some(stop_reason),
            usage: usage.map(Into::into),
        });
        out.push(StreamEvent::MessageStop);
        self.finished = true;
    }
}

/// Translate one parsed SSE event (name + JSON payload) into canonical
/// events, mutating `state`. Unknown event names are ignored — the API
/// adds event types over time and skipping them keeps the parser
/// forward-compatible.
pub fn process_event(event: &str, payload: &str, state: &mut StreamState) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    if state.finished {
        return out;
    }

    let event = resolve_event_name(event, payload);
    match event.as_ref() {
        "response.created" | "response.in_progress" => {
            if let Ok(env) = serde_json::from_str::<ResponseEnvelope>(payload) {
                if let Some(resp) = env.response {
                    if !state.started {
                        state.started = true;
                        out.push(StreamEvent::MessageStart {
                            message_id: resp.id.unwrap_or_default(),
                            model: resp.model,
                        });
                    }
                }
            }
        }
        "response.output_item.added" => {
            if let Ok(ev) = serde_json::from_str::<OutputItemEvent>(payload) {
                if let Some(OutputItem::FunctionCall { call_id, name, .. }) = ev.item {
                    let output_index = ev.output_index.unwrap_or_default();
                    match (
                        call_id.filter(|s| !s.is_empty()),
                        name.filter(|s| !s.is_empty()),
                    ) {
                        (Some(id), Some(name)) => {
                            state.block_for(
                                output_index,
                                BlockOpen::ToolUse { id, name },
                                &mut out,
                            );
                        }
                        // Same rule as the non-streaming converter: a tool
                        // call with no id/name can never be answered, so
                        // fail loudly instead of opening a block the engine
                        // would later fail to pair a tool_result against.
                        _ => state.fail("function_call item missing call_id or name", &mut out),
                    }
                }
            }
        }
        // Encrypted reasoning only arrives with the finished item, never
        // on `.added`. Attach it to the item's Thinking block — opening
        // one if no summary streamed — so the next request can replay it.
        "response.output_item.done" => {
            if let Ok(ev) = serde_json::from_str::<OutputItemEvent>(payload) {
                let output_index = ev.output_index.unwrap_or_default();
                match ev.item {
                    Some(OutputItem::Reasoning {
                        encrypted_content: Some(encrypted),
                        ..
                    }) if !encrypted.is_empty() => {
                        let idx = state.block_for(output_index, BlockOpen::Thinking, &mut out);
                        out.push(StreamEvent::ContentBlockDelta {
                            index: idx,
                            delta: ContentDelta::Signature {
                                signature: format!(
                                    "{}{encrypted}",
                                    super::REASONING_SIGNATURE_PREFIX
                                ),
                            },
                        });
                    }
                    // Some compatible endpoints send a call's arguments
                    // only on the finished item, with no argument deltas.
                    // Without this the tool would run with `{}` input.
                    Some(OutputItem::FunctionCall {
                        arguments: Some(arguments),
                        ..
                    }) if !arguments.is_empty() => {
                        if let Some(idx) = state.existing_block(output_index) {
                            if state.args_streamed.insert(idx) {
                                out.push(StreamEvent::ContentBlockDelta {
                                    index: idx,
                                    delta: ContentDelta::ToolInputJson {
                                        partial_json: arguments,
                                    },
                                });
                            }
                        }
                    }
                    _ => {}
                }
                // The item is finished: close its block now rather than at
                // `response.completed`, so the engine can pre-run a
                // read-only tool call while the rest of the reply streams.
                if let Some(idx) = state.existing_block(output_index) {
                    if state.open.remove(&idx) {
                        out.push(StreamEvent::ContentBlockStop { index: idx });
                    }
                }
            }
        }
        "response.output_text.delta" => {
            if let Ok(ev) = serde_json::from_str::<DeltaEvent>(payload) {
                if let Some(text) = ev.delta {
                    if !text.is_empty() {
                        let idx = state.block_for(
                            ev.output_index.unwrap_or_default(),
                            BlockOpen::Text,
                            &mut out,
                        );
                        out.push(StreamEvent::ContentBlockDelta {
                            index: idx,
                            delta: ContentDelta::Text { text },
                        });
                    }
                }
            }
        }
        // A refusal streams in place of output text. Surface it as text
        // with the same `[refused] ` prefix the non-streaming converter
        // uses — ignoring it would turn a refusal into an empty reply.
        "response.refusal.delta" => {
            if let Ok(ev) = serde_json::from_str::<DeltaEvent>(payload) {
                if let Some(delta) = ev.delta.filter(|d| !d.is_empty()) {
                    let output_index = ev.output_index.unwrap_or_default();
                    let text = if state.refusing.insert(output_index) {
                        format!("[refused] {delta}")
                    } else {
                        delta
                    };
                    let idx = state.block_for(output_index, BlockOpen::Text, &mut out);
                    out.push(StreamEvent::ContentBlockDelta {
                        index: idx,
                        delta: ContentDelta::Text { text },
                    });
                }
            }
        }
        // Reasoning models stream their chain-of-thought as summary or
        // raw reasoning text; surface both as canonical Thinking blocks.
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            if let Ok(ev) = serde_json::from_str::<DeltaEvent>(payload) {
                if let Some(mut text) = ev.delta {
                    if !text.is_empty() {
                        let output_index = ev.output_index.unwrap_or_default();
                        if let Some(part) = ev.summary_index {
                            if state
                                .summary_part
                                .insert(output_index, part)
                                .is_some_and(|prev| prev != part)
                            {
                                text.insert_str(0, "\n\n");
                            }
                        }
                        let idx = state.block_for(output_index, BlockOpen::Thinking, &mut out);
                        out.push(StreamEvent::ContentBlockDelta {
                            index: idx,
                            delta: ContentDelta::Thinking { text },
                        });
                    }
                }
            }
        }
        "response.function_call_arguments.delta" => {
            if let Ok(ev) = serde_json::from_str::<DeltaEvent>(payload) {
                if let Some(args) = ev.delta {
                    if !args.is_empty() {
                        // Arguments can only attach to a tool block that
                        // `response.output_item.added` already opened —
                        // that is the sole event carrying `call_id` /
                        // `name`. Opening one here would have to invent an
                        // empty id, so a missing block is a protocol
                        // violation rather than something to paper over.
                        match state.existing_block(ev.output_index.unwrap_or_default()) {
                            Some(idx) => {
                                state.args_streamed.insert(idx);
                                out.push(StreamEvent::ContentBlockDelta {
                                    index: idx,
                                    delta: ContentDelta::ToolInputJson { partial_json: args },
                                })
                            }
                            None => state.fail(
                                "function_call_arguments delta with no preceding \
                                 output_item.added",
                                &mut out,
                            ),
                        }
                    }
                }
            }
        }
        "response.completed" | "response.incomplete" => {
            if let Ok(env) = serde_json::from_str::<ResponseEnvelope>(payload) {
                let resp = env.response.unwrap_or_default();
                if !state.started {
                    state.started = true;
                    out.push(StreamEvent::MessageStart {
                        message_id: resp.id.clone().unwrap_or_default(),
                        model: resp.model.clone(),
                    });
                }
                state.finish(
                    resp.status.as_deref(),
                    resp.incomplete_details.as_ref(),
                    resp.usage,
                    &mut out,
                );
            }
        }
        "response.failed" => {
            if let Ok(env) = serde_json::from_str::<ResponseEnvelope>(payload) {
                let message = env
                    .response
                    .and_then(|r| r.error)
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "response failed".into());
                state.fail(message, &mut out);
            }
        }
        "error" => {
            if let Ok(ev) = serde_json::from_str::<ErrorEvent>(payload) {
                state.fail(ev.describe(), &mut out);
            }
        }
        // response.content_part.* / response.output_text.done /
        // response.function_call_arguments.done / … — bookkeeping we don't
        // need.
        _ => {}
    }

    out
}

/// Resolve an SSE frame's event name. Frames normally carry an explicit
/// `event:` line, but some proxies strip it; the Responses API's data
/// payloads then still identify themselves via a `type` field mirroring
/// the event name, so fall back to that.
fn resolve_event_name<'a>(event: &'a str, payload: &str) -> std::borrow::Cow<'a, str> {
    if !event.is_empty() {
        return std::borrow::Cow::Borrowed(event);
    }
    let derived = serde_json::from_str::<serde_json::Value>(payload)
        .ok()
        .and_then(|v| v.get("type")?.as_str().map(str::to_string))
        .unwrap_or_default();
    std::borrow::Cow::Owned(derived)
}

/// Drive a byte stream through SSE frame buffering and emit canonical
/// [`StreamEvent`]s. A frame is an optional `event:` line plus one or
/// more `data:` lines (joined with `\n` per the SSE spec), terminated by
/// a blank line. Dispatches a trailing unterminated frame at end of
/// stream. Completion is decided by what was seen, not by the body ending:
/// a terminal event (`response.completed` / `.incomplete` / `.failed`)
/// ends the reply; a `data: [DONE]` sentinel (some proxies send it instead)
/// after output began ends it as `response.completed` would; anything
/// else — a body cut mid-reply, or one that produced no output (empty,
/// keep-alive only, or `response.created` then `[DONE]`) — is a
/// `StreamInterrupted`, so the engine retries.
pub fn parse_byte_stream<S>(input: S) -> BoxStream<'static, LlmResult<StreamEvent>>
where
    S: Stream<Item = LlmResult<Bytes>> + Send + 'static,
{
    let stream = async_stream::try_stream! {
        let mut buf = crate::transport::SseLineBuffer::default();
        let mut state = StreamState::new();
        let mut input = std::pin::pin!(input);
        let mut pending_event = String::new();
        let mut data_lines: Vec<String> = Vec::new();
        // An explicit end marker some proxies send instead of (or after)
        // `response.completed`.
        let mut saw_done = false;

        'outer: while let Some(chunk) = input.next().await {
            let chunk = chunk?;
            buf.push(&chunk);

            while let Some(line) = buf.next_line() {
                if line.is_empty() {
                    // Blank line = end of frame → dispatch.
                    if !data_lines.is_empty() {
                        let event = std::mem::take(&mut pending_event);
                        let payload = data_lines.join("\n");
                        data_lines.clear();
                        for ev in process_event(&event, &payload, &mut state) {
                            yield ev;
                        }
                    }
                    continue;
                }
                if let Some(name) = line.strip_prefix("event:") {
                    pending_event = name.trim().to_string();
                    continue;
                }
                let Some(payload) = line.strip_prefix("data:") else {
                    continue; // comments / retry lines / unknown fields
                };
                // Spec: a single optional space after the colon is removed.
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if payload == "[DONE]" {
                    saw_done = true;
                    break 'outer;
                }
                data_lines.push(payload.to_string());
            }
        }

        // End of stream terminates the final line (SSE spec): classify
        // any residual line so a truncated frame still dispatches.
        if let Some(line) = buf.take_rest() {
            let line = line.as_str();
            if let Some(name) = line.strip_prefix("event:") {
                pending_event = name.trim().to_string();
            } else if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if payload == "[DONE]" {
                    saw_done = true;
                } else {
                    data_lines.push(payload.to_string());
                }
            }
        }

        // A trailing frame without its blank-line terminator still counts.
        if !data_lines.is_empty() {
            let event = std::mem::take(&mut pending_event);
            let payload = data_lines.join("\n");
            for ev in process_event(&event, &payload, &mut state) {
                yield ev;
            }
        }

        if !state.finished {
            if saw_done && state.next_index > 0 {
                // `[DONE]` in place of `response.completed`: finish the
                // same way, closing open blocks (the engine flushes text it
                // held back on each block's stop) and inferring the stop
                // reason.
                let mut out = Vec::new();
                state.finish(None, None, None, &mut out);
                for ev in out {
                    yield ev;
                }
            } else {
                // No terminal event: the connection dropped mid-reply, or
                // nothing was produced — HTTP 200 with an empty or
                // keep-alive-only body, or a relay that opened the response
                // and then swallowed the error. Neither is a finished reply;
                // surfacing an interruption lets the engine retry instead of
                // keeping half an answer, or ending the turn with nothing.
                Err(LlmError::StreamInterrupted(
                    "Responses stream ended before response.completed".into(),
                ))?;
            }
        }
    };
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LlmError;
    use crate::response::StopReason;
    use serde_json::json;
    use snaca_core::ContentBlock;

    fn process_value(
        event: &str,
        value: serde_json::Value,
        state: &mut StreamState,
    ) -> Vec<StreamEvent> {
        process_event(event, &value.to_string(), state)
    }

    fn tool_start(events: &[StreamEvent]) -> Option<(String, String)> {
        events.iter().find_map(|e| match e {
            StreamEvent::ContentBlockStart {
                block: ContentBlockStart::ToolUse { id, name },
                ..
            } => Some((id.clone(), name.clone())),
            _ => None,
        })
    }

    #[test]
    fn refusal_deltas_surface_as_prefixed_text() {
        // Without this: a streamed refusal finishes as an empty EndTurn,
        // while the non-streaming path returns `[refused] …`.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1", "model": "gpt-5-mini"}}),
            &mut state,
        );
        let mut text = String::new();
        for delta in ["I can't", " help with that"] {
            for ev in process_value(
                "response.refusal.delta",
                json!({"output_index": 0, "delta": delta}),
                &mut state,
            ) {
                if let StreamEvent::ContentBlockDelta {
                    delta: ContentDelta::Text { text: t },
                    ..
                } = ev
                {
                    text.push_str(&t);
                }
            }
        }
        assert_eq!(text, "[refused] I can't help with that");
    }

    #[test]
    fn function_call_item_without_call_id_fails_the_stream() {
        // Without this: defaulting a missing id/name opens a tool block
        // with an empty id — which breaks the engine's tool_use/tool_result
        // pairing and misroutes the eager prerun's result. Fail like the
        // non-streaming converter instead.
        for item in [
            json!({"type": "function_call", "name": "Read", "arguments": ""}),
            json!({"type": "function_call", "call_id": "", "name": "Read", "arguments": ""}),
            json!({"type": "function_call", "call_id": "call_1", "arguments": ""}),
            json!({"type": "function_call", "call_id": "call_1", "name": "", "arguments": ""}),
        ] {
            let mut state = StreamState::new();
            let evs = process_value(
                "response.output_item.added",
                json!({"output_index": 1, "item": item}),
                &mut state,
            );
            assert!(
                tool_start(&evs).is_none(),
                "must not open a tool block: {evs:?}"
            );
            assert!(matches!(
                &evs[0],
                StreamEvent::Error { message } if message.contains("missing call_id or name")
            ));
            // Terminal — later frames are ignored.
            assert!(process_value(
                "response.output_text.delta",
                json!({"output_index": 0, "delta": "Hi"}),
                &mut state,
            )
            .is_empty());
        }
    }

    #[test]
    fn orphan_function_call_args_delta_fails_the_stream() {
        // Without this: an argument delta with no `output_item.added` (the
        // only event carrying call_id / name) opened a tool block with an
        // empty id, breaking tool_use/tool_result pairing and getting the
        // replayed history rejected.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1"}}),
            &mut state,
        );
        let evs = process_value(
            "response.function_call_arguments.delta",
            json!({"output_index": 7, "delta": "{\"path\":"}),
            &mut state,
        );
        assert!(
            tool_start(&evs).is_none(),
            "must not open a tool block: {evs:?}"
        );
        assert!(matches!(
            &evs[0],
            StreamEvent::Error { message } if message.contains("no preceding output_item.added")
        ));
    }

    #[test]
    fn completed_emits_usage_and_stop() {
        // Without this: `response.completed` dropped usage (token stats read
        // 0 and auto-compaction never fired) or left open blocks unclosed.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1"}}),
            &mut state,
        );
        process_value(
            "response.output_text.delta",
            json!({"output_index": 0, "delta": "Hi"}),
            &mut state,
        );
        let evs = process_value(
            "response.completed",
            json!({"response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {"input_tokens": 5, "output_tokens": 2,
                          "input_tokens_details": {"cached_tokens": 4}}
            }}),
            &mut state,
        );
        // ContentBlockStop(0) + MessageDelta + MessageStop
        assert_eq!(evs.len(), 3, "got {evs:#?}");
        assert!(matches!(evs[0], StreamEvent::ContentBlockStop { index: 0 }));
        match &evs[1] {
            StreamEvent::MessageDelta { stop_reason, usage } => {
                assert_eq!(*stop_reason, Some(StopReason::EndTurn));
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input_tokens, 5);
                assert_eq!(u.cache_read_input_tokens, Some(4));
            }
            other => panic!("got {other:?}"),
        }
        assert!(matches!(evs[2], StreamEvent::MessageStop));
        assert!(state.finished);
    }

    #[test]
    fn completed_with_tool_call_reports_tool_use() {
        // Without this: OpenAI closes a call's block on `output_item.done`
        // before `response.completed`; counting only still-open blocks, the
        // reply finished as EndTurn and the engine ended the turn without
        // running the tool.
        let mut state = StreamState::new();
        for (name, payload) in [
            ("response.created", json!({"response": {"id": "resp_1"}})),
            (
                "response.output_item.added",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "c1", "name": "Bash", "arguments": ""}}),
            ),
            (
                "response.output_item.done",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "c1", "name": "Bash", "arguments": "{}"}}),
            ),
        ] {
            process_value(name, payload, &mut state);
        }
        let evs = process_value(
            "response.completed",
            json!({"response": {"id": "resp_1", "status": "completed",
                                "usage": {"input_tokens": 9, "output_tokens": 3}}}),
            &mut state,
        );
        assert!(
            matches!(
                evs.as_slice(),
                [
                    StreamEvent::MessageDelta {
                        stop_reason: Some(StopReason::ToolUse),
                        ..
                    },
                    StreamEvent::MessageStop,
                ]
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn incomplete_maps_to_max_tokens() {
        // Without this: a reply cut off at max_output_tokens never finished
        // (or finished as EndTurn), so the engine retried or kept the
        // truncated text instead of raising the output cap and continuing.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1"}}),
            &mut state,
        );
        let evs = process_value(
            "response.incomplete",
            json!({"response": {"id": "resp_1", "status": "incomplete",
                                "incomplete_details": {"reason": "max_output_tokens"}}}),
            &mut state,
        );
        // No open blocks → just [MessageDelta, MessageStop].
        match &evs[0] {
            StreamEvent::MessageDelta { stop_reason, .. } => {
                assert_eq!(*stop_reason, Some(StopReason::MaxTokens));
            }
            other => panic!("got {other:?}"),
        }
        assert!(matches!(evs[1], StreamEvent::MessageStop));
    }

    #[test]
    fn reasoning_deltas_surface_as_thinking_block() {
        // Without this: reasoning — OpenAI's summaries, or the raw reasoning
        // text open-weight / compatible endpoints stream instead — leaked
        // into the visible reply or was dropped instead of landing in a
        // Thinking block.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1"}}),
            &mut state,
        );
        for (output_index, event) in [
            (0, "response.reasoning_summary_text.delta"),
            (1, "response.reasoning_text.delta"),
        ] {
            let evs = process_value(
                event,
                json!({"output_index": output_index, "delta": "thinking hard"}),
                &mut state,
            );
            assert!(
                matches!(
                    evs.as_slice(),
                    [
                        StreamEvent::ContentBlockStart {
                            block: ContentBlockStart::Thinking,
                            ..
                        },
                        StreamEvent::ContentBlockDelta {
                            delta: ContentDelta::Thinking { text },
                            ..
                        },
                    ] if text == "thinking hard"
                ),
                "{event}: {evs:?}"
            );
        }
    }

    #[test]
    fn reasoning_item_done_attaches_encrypted_signature() {
        // Without this: encrypted reasoning from `output_item.done` never
        // reached history (always, when no summary streamed), so the tool
        // loop could not replay a reasoning model's chain of thought.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1", "model": "gpt-5-mini"}}),
            &mut state,
        );
        // Summary streamed first: the signature lands on that block.
        process_value(
            "response.reasoning_summary_text.delta",
            json!({"output_index": 0, "delta": "plan"}),
            &mut state,
        );
        let evs = process_value(
            "response.output_item.done",
            json!({"output_index": 0, "item": {
                "type": "reasoning", "id": "rs_1", "summary": [],
                "encrypted_content": "gAAA"}}),
            &mut state,
        );
        assert!(matches!(
            evs.as_slice(),
            [
                StreamEvent::ContentBlockDelta { index: 0, delta: ContentDelta::Signature { signature } },
                StreamEvent::ContentBlockStop { index: 0 },
            ] if signature == "openai-responses:gAAA"
        ));
        // No summary at all: the done item still opens a Thinking block,
        // or the encrypted reasoning would be lost.
        let evs = process_value(
            "response.output_item.done",
            json!({"output_index": 1, "item": {
                "type": "reasoning", "id": "rs_2", "summary": [],
                "encrypted_content": "gBBB"}}),
            &mut state,
        );
        assert!(matches!(
            &evs[0],
            StreamEvent::ContentBlockStart {
                index: 1,
                block: ContentBlockStart::Thinking
            }
        ));
        assert!(matches!(
            &evs[1],
            StreamEvent::ContentBlockDelta { index: 1, delta: ContentDelta::Signature { signature } }
                if signature == "openai-responses:gBBB"
        ));
        assert!(matches!(
            &evs[2],
            StreamEvent::ContentBlockStop { index: 1 }
        ));
    }

    #[test]
    fn finished_item_closes_its_block_before_the_response_ends() {
        // Without this: blocks close only on `response.completed`, so the
        // engine can't pre-run a read-only call while the rest of the
        // reply streams — the prerun win other providers get is lost.
        let mut state = StreamState::new();
        let mut events = Vec::new();
        for (name, payload) in [
            (
                "response.created",
                json!({"response": {"id": "r", "model": "m"}}),
            ),
            (
                "response.output_item.added",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read"}}),
            ),
            (
                "response.function_call_arguments.delta",
                json!({"output_index": 0, "delta": "{}"}),
            ),
            (
                "response.output_item.done",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"}}),
            ),
        ] {
            events.extend(process_value(name, payload, &mut state));
        }
        assert!(matches!(
            events.last(),
            Some(StreamEvent::ContentBlockStop { index: 0 })
        ));
        // …and the terminal event doesn't close it a second time.
        let evs = process_value(
            "response.completed",
            json!({"response": {"id": "r", "status": "completed"}}),
            &mut state,
        );
        assert!(!evs
            .iter()
            .any(|e| matches!(e, StreamEvent::ContentBlockStop { .. })));
    }

    #[test]
    fn summary_parts_are_separated_like_the_non_streaming_path() {
        // Without this: streaming joins "Plan A." and "Check B." into
        // "Plan A.Check B." while the same response parsed non-streaming
        // reads "Plan A.\n\nCheck B.".
        let mut state = StreamState::new();
        let mut text = String::new();
        for (part, delta) in [(0, "Plan A."), (1, "Check B.")] {
            for ev in process_value(
                "response.reasoning_summary_text.delta",
                json!({"output_index": 0, "summary_index": part, "delta": delta}),
                &mut state,
            ) {
                if let StreamEvent::ContentBlockDelta {
                    delta: ContentDelta::Thinking { text: t },
                    ..
                } = ev
                {
                    text.push_str(&t);
                }
            }
        }
        assert_eq!(text, "Plan A.\n\nCheck B.");
    }

    fn tool_args(events: &[StreamEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: ContentDelta::ToolInputJson { partial_json },
                    ..
                } => Some(partial_json.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn function_call_args_fall_back_to_the_finished_item() {
        // Without this: an endpoint that sends a call's arguments only on
        // `output_item.done` (no argument deltas) yields a tool call with
        // `{}` input — e.g. Read with no path.
        let mut state = StreamState::new();
        let mut events = Vec::new();
        for (name, payload) in [
            (
                "response.created",
                json!({"response": {"id": "r", "model": "m"}}),
            ),
            (
                "response.output_item.added",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}}),
            ),
            (
                "response.output_item.done",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read",
                "arguments": "{\"path\":\"x\"}"}}),
            ),
        ] {
            events.extend(process_value(name, payload, &mut state));
        }
        assert_eq!(tool_args(&events), "{\"path\":\"x\"}");
    }

    #[test]
    fn streamed_function_call_args_are_not_duplicated_by_done() {
        // Without this: the fallback would append the full arguments after
        // the streamed ones, producing invalid JSON (`{..}{..}`).
        let mut state = StreamState::new();
        let mut events = Vec::new();
        for (name, payload) in [
            (
                "response.created",
                json!({"response": {"id": "r", "model": "m"}}),
            ),
            (
                "response.output_item.added",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read"}}),
            ),
            (
                "response.function_call_arguments.delta",
                json!({"output_index": 0, "delta": "{\"path\":\"x\"}"}),
            ),
            (
                "response.output_item.done",
                json!({"output_index": 0, "item": {
                "type": "function_call", "call_id": "call_1", "name": "Read",
                "arguments": "{\"path\":\"x\"}"}}),
            ),
        ] {
            events.extend(process_value(name, payload, &mut state));
        }
        assert_eq!(tool_args(&events), "{\"path\":\"x\"}");
    }

    #[test]
    fn failed_event_yields_error() {
        // Without this: `response.failed` ended the stream as a retried
        // interruption, or surfaced as a bare "response failed" without the
        // provider's code and message.
        let mut state = StreamState::new();
        process_value(
            "response.created",
            json!({"response": {"id": "resp_1"}}),
            &mut state,
        );
        let evs = process_value(
            "response.failed",
            json!({"response": {"id": "resp_1", "status": "failed",
                                "error": {"code": "server_error", "message": "boom"}}}),
            &mut state,
        );
        assert!(matches!(&evs[0], StreamEvent::Error { message } if message.contains("boom")));
        assert!(state.finished);
    }

    #[test]
    fn unknown_events_are_ignored() {
        // Without this: bookkeeping events OpenAI sends in every stream
        // (content_part.added, output_text.done, …) could fail or end it,
        // breaking every reply.
        let mut state = StreamState::new();
        let evs = process_value(
            "response.content_part.added",
            json!({"output_index": 0, "content_index": 0}),
            &mut state,
        );
        assert!(evs.is_empty());
        assert!(!state.finished);
    }

    fn collect_events(bytes: &[u8]) -> Vec<StreamEvent> {
        use futures::executor::block_on;
        use futures::stream;
        let owned: Vec<u8> = bytes.to_vec();
        let s = stream::once(async move { Ok::<_, LlmError>(Bytes::from(owned)) });
        block_on(async {
            parse_byte_stream(s)
                .filter_map(|r| async { r.ok() })
                .collect::<Vec<_>>()
                .await
        })
    }

    #[test]
    fn end_to_end_byte_stream() {
        // Without this: a complete stream ending in `response.completed`
        // could still end in StreamInterrupted, so a finished reply was
        // retried — or each text delta opened a new block, splitting it.
        let raw = b"\
event: response.created\n\
data: {\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\"}}\n\
\n\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hi\"}\n\
\n\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\" there\"}\n\
\n\
event: response.completed\n\
data: {\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}\n\
\n";
        let events = collect_results(raw)
            .into_iter()
            .collect::<LlmResult<Vec<_>>>()
            .expect("a completed stream yields no error");
        // start, block_start, delta, delta, block_stop, message_delta, stop
        assert_eq!(events.len(), 7, "got {events:#?}");
        assert!(matches!(
            &events[0],
            StreamEvent::MessageStart { message_id, .. } if message_id == "resp_1"
        ));
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentBlockDelta {
                    delta: ContentDelta::Text { text },
                    ..
                } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hi there");
        assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
    }

    #[test]
    fn split_chunk_mid_frame_resumes_correctly() {
        // Without this: an SSE line split across two network chunks lost its
        // first half, silently dropping deltas whenever TCP split mid-line.
        use futures::executor::block_on;
        use futures::stream;

        let part_a = b"event: response.output_text.delta\n\
                       data: {\"output_index\":0,\"del";
        let part_b = b"ta\":\"Hi\"}\n\n";
        let s = stream::iter(vec![
            Ok::<_, LlmError>(Bytes::from(part_a.to_vec())),
            Ok(Bytes::from(part_b.to_vec())),
        ]);
        let events: Vec<StreamEvent> = block_on(async {
            parse_byte_stream(s)
                .filter_map(|r| async { r.ok() })
                .collect()
                .await
        });
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Text { text },
                ..
            } if text == "Hi"
        )));
        // Stream ended without `response.created` ever being seen, so
        // the state machine never "started" — no synthetic MessageStop.
        assert!(!matches!(events.last(), Some(StreamEvent::MessageStop)));
    }

    /// Data-only frames (no `event:` line) — some proxies strip event
    /// names. The payload's `type` field mirrors the event name on the
    /// Responses API, so routing still works.
    #[test]
    fn data_only_frames_route_via_type_field() {
        // Without this: behind a proxy that strips `event:` lines every
        // frame was ignored, so the reply streamed nothing and failed as
        // interrupted.
        let raw = b"\
data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5-mini\"}}\n\
\n\
data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hi\"}\n\
\n";
        let events = collect_events(raw);
        assert!(matches!(
            &events[0],
            StreamEvent::MessageStart { message_id, .. } if message_id == "resp_1"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::ContentBlockStart {
                block: ContentBlockStart::Text,
                ..
            }
        ));
        assert!(matches!(
            &events[2],
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Text { text },
                ..
            } if text == "Hi"
        ));
    }

    fn collect_results(bytes: &[u8]) -> Vec<LlmResult<StreamEvent>> {
        use futures::executor::block_on;
        use futures::stream;
        let owned: Vec<u8> = bytes.to_vec();
        let s = stream::once(async move { Ok::<_, LlmError>(Bytes::from(owned)) });
        block_on(parse_byte_stream(s).collect::<Vec<_>>())
    }

    #[test]
    fn stream_cut_before_the_terminal_event_is_an_interruption() {
        // Without this: a dropped connection ends the stream mid-answer
        // with no error and no stop_reason, so half a reply (or half a
        // tool call) is kept as a normal end of turn and never retried.
        let raw = b"\
event: response.created\n\
data: {\"response\":{\"id\":\"r\",\"model\":\"m\"}}\n\
\n\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hal\"}\n\
\n";
        let results = collect_results(raw);
        assert!(
            matches!(results.last(), Some(Err(LlmError::StreamInterrupted(_)))),
            "{results:?}"
        );
    }

    #[test]
    fn stream_that_never_starts_is_an_interruption() {
        // Without this: HTTP 200 with an empty body, or one carrying only
        // keep-alive comments, parses as zero events and zero errors; the
        // accumulator then reports an empty EndTurn and the engine ends the
        // turn silently — no reply, no retry.
        // A bare `[DONE]`, or one right after `response.created` (a relay
        // that opened the response and swallowed the error), ends nothing.
        for raw in [
            &b""[..],
            &b": keep-alive\n\n: keep-alive\n\n"[..],
            &b"data: [DONE]\n\n"[..],
            &b"event: response.created\ndata: {\"response\":{\"id\":\"r\"}}\n\ndata: [DONE]\n\n"[..],
        ] {
            let results = collect_results(raw);
            assert!(
                matches!(
                    results.as_slice(),
                    [.., Err(LlmError::StreamInterrupted(_))]
                ) && results.iter().all(|r| !matches!(
                    r,
                    Ok(StreamEvent::ContentBlockDelta { .. } | StreamEvent::MessageStop)
                )),
                "{:?} -> {results:?}",
                String::from_utf8_lossy(raw)
            );
        }
    }

    #[test]
    fn done_marker_closes_open_blocks_like_completed() {
        // Without this: a `[DONE]`-terminated stream ends with no
        // `ContentBlockStop`, and the engine — which flushes the text its
        // fence scrubber held back only on a block's stop — loses the end
        // of the reply; the stop reason is never set either.
        let raw = b"\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hi\"}\n\
\n\
data: [DONE]\n\
\n";
        let events = collect_events(raw);
        let tail: Vec<_> = events.iter().rev().take(3).rev().collect();
        assert!(
            matches!(
                tail.as_slice(),
                [
                    StreamEvent::ContentBlockStop { .. },
                    StreamEvent::MessageDelta {
                        stop_reason: Some(StopReason::EndTurn),
                        ..
                    },
                    StreamEvent::MessageStop,
                ]
            ),
            "{events:?}"
        );
    }

    #[test]
    fn nested_error_event_keeps_its_code_and_message() {
        // Without this: an `error` event with its details under `error`
        // surfaced as a bare "stream error", hiding why the turn failed.
        let mut state = StreamState::new();
        let evs = process_value(
            "error",
            json!({"type": "error", "error": {"code": "server_error", "message": "upstream boom"}}),
            &mut state,
        );
        assert!(
            matches!(&evs[0], StreamEvent::Error { message } if message == "server_error: upstream boom"),
            "{evs:?}"
        );
        let evs = process_value(
            "error",
            json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}),
            &mut StreamState::new(),
        );
        assert!(
            matches!(&evs[0], StreamEvent::Error { message } if message == "rate_limit_exceeded: slow down")
        );
    }

    #[test]
    fn done_marker_after_content_without_created_still_ends_cleanly() {
        // Without this: a proxy that ends streams with `[DONE]` instead of
        // `response.completed` (with or without `response.created`) streams
        // a full reply that then fails as interrupted — the engine retries,
        // and every retry fails the same way.
        for raw in [
            &b"\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hi\"}\n\
\n\
data: [DONE]\n\
\n"[..],
            &b"\
event: response.created\n\
data: {\"response\":{\"id\":\"r\",\"model\":\"m\"}}\n\
\n\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hi\"}\n\
\n\
data: [DONE]\n\
\n"[..],
        ] {
            let results = collect_results(raw);
            assert!(results.iter().all(Result::is_ok), "{results:?}");
            assert!(matches!(results.last(), Some(Ok(StreamEvent::MessageStop))));
        }
    }

    /// A JSON payload split across multiple `data:` lines is joined with
    /// `\n` per the SSE spec (multi-line data), which stays valid JSON.
    #[test]
    fn multi_line_data_frames_are_joined() {
        // Without this: a payload split over several `data:` lines (legal
        // SSE, sent by some proxies) kept only one line and was dropped as
        // invalid JSON.
        let raw = b"\
event: response.output_text.delta\n\
data: {\"output_index\":0,\n\
data: \"delta\":\"Hi\"}\n\
\n\
data: [DONE]\n\
\n";
        let events = collect_events(raw);
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Text { text },
                ..
            } if text == "Hi"
        )));
    }

    /// A frame without its blank-line terminator still dispatches when
    /// the stream ends (truncated proxies).
    #[test]
    fn trailing_frame_without_blank_line_still_dispatches() {
        // Without this: a server that ends the body right after the last
        // `data:` line (no blank line) lost that frame, so a completed reply
        // was reported as interrupted and retried.
        let raw = b"\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hi\"}\n\
\n\
event: response.completed\n\
data: {\"response\":{\"id\":\"r\",\"status\":\"completed\"}}";
        let results = collect_results(raw);
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        assert!(matches!(results.last(), Some(Ok(StreamEvent::MessageStop))));
    }

    #[test]
    fn stream_accumulates_into_response() {
        // Without this: streamed events stopped reassembling into the
        // response the engine consumes — e.g. argument deltas landing on the
        // wrong block — so the tool ran with `{}` input.
        let mut acc = crate::stream::StreamAccumulator::new();
        let mut state = StreamState::new();
        let events = {
            let mut v = Vec::new();
            v.extend(process_value(
                "response.created",
                json!({"response": {"id": "resp_9", "model": "gpt-5-mini"}}),
                &mut state,
            ));
            v.extend(process_value(
                "response.output_text.delta",
                json!({"output_index": 0, "delta": "Hello"}),
                &mut state,
            ));
            v.extend(process_value(
                "response.output_item.added",
                json!({"output_index": 1, "item": {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": ""}}),
                &mut state,
            ));
            v.extend(process_value(
                "response.function_call_arguments.delta",
                json!({"output_index": 1, "delta": "{\"path\":\"x\"}"}),
                &mut state,
            ));
            v.extend(process_value(
                "response.completed",
                json!({"response": {"id": "resp_9", "status": "completed",
                                    "usage": {"input_tokens": 7, "output_tokens": 4}}}),
                &mut state,
            ));
            v
        };
        for ev in events {
            acc.ingest(ev);
        }
        let resp = acc.finalize().unwrap();
        assert_eq!(resp.id, "resp_9");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 7);
        assert_eq!(resp.usage.output_tokens, 4);
        assert_eq!(resp.message.content.len(), 2);
        assert!(matches!(
            &resp.message.content[0],
            ContentBlock::Text { text } if text == "Hello"
        ));
        match &resp.message.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id.as_str(), "call_1");
                assert_eq!(name, "Read");
                assert_eq!(input, &json!({"path": "x"}));
            }
            other => panic!("got {other:?}"),
        }
    }
}

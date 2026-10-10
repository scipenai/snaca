//! Canonical ↔ OpenAI Responses API conversion.
//!
//! ## Outbound (request) translation
//! - System prompt (top-level + history `Role::System` messages) →
//!   top-level `instructions`.
//! - `Role::User` → `message` item with `input_text` / `input_image`
//!   parts. Base64 images become `data:` URLs; with vision off they
//!   degrade to `[image omitted: …]` placeholder text.
//! - `Role::Assistant` → `message` item with `output_text` (replayed in
//!   the API's own output shape) plus one `function_call` item per
//!   `ToolUse` block. In the current tool loop (after the last user
//!   message), `Thinking` blocks carrying this provider's encrypted
//!   reasoning are replayed as id-less `reasoning` items, each directly
//!   before the item it produced (see `REASONING_SIGNATURE_PREFIX`).
//! - `Role::Tool` → one `function_call_output` item per `ToolResult`,
//!   its output a string. Tool-returned images are re-surfaced as a
//!   follow-up user message: OpenAI also takes `input_image` parts inside
//!   the output, but compatible gateways may not, and a user message
//!   works everywhere.
//! - `stop_sequences` is ignored — the Responses API has no stop
//!   parameter.

use crate::error::{LlmError, LlmResult};
use crate::images::{image_url_of, split_images, tool_images_message_parts, ImagePolicy};
use crate::openai_responses::wire::{
    InputContentPart, InputItem, InputSummaryPart, OutputContentPart, OutputItem,
    OutputSummaryPart, ReasoningParam, ResponsesRequest, ResponsesResponse, ResponsesTool,
};
use crate::openai_responses::REASONING_SIGNATURE_PREFIX;
use crate::request::MessageRequest;
use crate::response::{MessageResponse, StopReason};
use snaca_core::{
    current_turn_start, ContentBlock, ImageSource, Message, MessageId, Role, ToolUseId,
};
use std::time::SystemTime;

/// `max_output_tokens` sent when the request sets none, as the Anthropic
/// converter does — an uncapped reasoning model can run long.
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

/// Smallest `max_output_tokens` the API accepts ("must be greater than or
/// equal to 16"). The engine's content-filter probe asks for 1 token;
/// sent as-is it fails with a 400 the probe can't read, and a flagged
/// history message is never found and redacted.
const MIN_MAX_OUTPUT_TOKENS: u32 = 16;

/// How a request handles reasoning.
#[derive(Debug, Clone, Default)]
pub(super) struct ReasoningMode {
    /// Ask for `reasoning.encrypted_content` (reasoning models only).
    pub include_encrypted: bool,
    /// Replay stored encrypted reasoning within the current tool loop.
    pub replay: bool,
    /// `reasoning.effort`, when configured.
    pub effort: Option<String>,
}

/// Request with the default reasoning handling: replay on, no `include`
/// (the client decides `include` per model).
#[cfg(test)]
pub fn build_responses_request(
    req: &MessageRequest,
    stream: bool,
    supports_vision: bool,
) -> LlmResult<ResponsesRequest> {
    build_responses_request_with(
        req,
        stream,
        supports_vision,
        ReasoningMode {
            replay: true,
            ..ReasoningMode::default()
        },
    )
}

pub(super) fn build_responses_request_with(
    req: &MessageRequest,
    stream: bool,
    supports_vision: bool,
    reasoning: ReasoningMode,
) -> LlmResult<ResponsesRequest> {
    let mut instructions_parts: Vec<String> = Vec::new();
    if let Some(system) = req.flat_system() {
        if !system.trim().is_empty() {
            instructions_parts.push(system);
        }
    }

    // OpenAI wants the reasoning that produced the pending function calls
    // replayed; earlier turns' reasoning is ignored by the model, so only
    // the current tool loop carries it.
    let replay_from = current_turn_start(&req.messages);
    let mut input: Vec<InputItem> = Vec::with_capacity(req.messages.len());
    for (i, msg) in req.messages.iter().enumerate() {
        match msg.role {
            Role::System => {
                let text = ContentBlock::collect_text(&msg.content);
                if !text.trim().is_empty() {
                    instructions_parts.push(text);
                }
            }
            Role::User => {
                input.push(user_message_item(msg, supports_vision));
            }
            Role::Assistant => {
                push_assistant_items(msg, &mut input, reasoning.replay && i >= replay_from)?;
            }
            Role::Tool => {
                push_tool_output_items(msg, &mut input, supports_vision);
            }
        }
    }

    let instructions = if instructions_parts.is_empty() {
        None
    } else {
        Some(instructions_parts.join("\n\n"))
    };

    let tools = req
        .tools
        .iter()
        .map(|s| ResponsesTool {
            tool_type: "function",
            name: s.name.clone(),
            description: s.description.clone(),
            parameters: s.input_schema.clone(),
            strict: false,
        })
        .collect();

    Ok(ResponsesRequest {
        model: req.model.clone(),
        instructions,
        input,
        tools,
        max_output_tokens: Some(
            req.max_tokens
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
                .max(MIN_MAX_OUTPUT_TOKENS),
        ),
        temperature: req.temperature,
        stream,
        store: false,
        include: if reasoning.include_encrypted {
            vec!["reasoning.encrypted_content"]
        } else {
            Vec::new()
        },
        reasoning: reasoning.effort.map(|effort| ReasoningParam { effort }),
    })
}

fn input_image(source: &ImageSource) -> InputContentPart {
    InputContentPart::InputImage {
        image_url: image_url_of(source),
        detail: None,
    }
}

fn user_message_item(msg: &Message, supports_vision: bool) -> InputItem {
    let (text, mut content) = split_images(
        &msg.content,
        ImagePolicy::openai(supports_vision),
        input_image,
    );
    if !text.is_empty() || content.is_empty() {
        // Always emit a text part (even empty) for text-only messages, the
        // shape the other converters use; an image-only vision message
        // stays image-only.
        content.insert(0, InputContentPart::InputText { text });
    }
    InputItem::Message {
        role: "user".into(),
        content,
    }
}

/// Assistant items in canonical block order: each run of Text blocks is one
/// `message`, each ToolUse a `function_call`. With `replay_reasoning`, a
/// Thinking block holding this provider's encrypted reasoning becomes an
/// id-less `reasoning` item, kept directly before the `message` /
/// `function_call` it produced — the API rejects a reasoning item with no
/// item right after it.
fn push_assistant_items(
    msg: &Message,
    out: &mut Vec<InputItem>,
    replay_reasoning: bool,
) -> LlmResult<()> {
    let mut items: Vec<InputItem> = Vec::new();
    // Consecutive Text blocks form one message, joined like `collect_text`.
    let mut text = String::new();
    let flush_text = |text: &mut String, items: &mut Vec<InputItem>| {
        if !text.is_empty() {
            items.push(InputItem::Message {
                role: "assistant".into(),
                content: vec![InputContentPart::OutputText {
                    text: std::mem::take(text),
                }],
            });
        }
    };
    for block in &msg.content {
        match block {
            ContentBlock::Text { text: t } => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            ContentBlock::ToolUse { id, name, input } => {
                flush_text(&mut text, &mut items);
                items.push(InputItem::FunctionCall {
                    call_id: id.as_str().to_string(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input)?,
                });
            }
            other if replay_reasoning => {
                if let Some(reasoning) = replayable_reasoning(other) {
                    flush_text(&mut text, &mut items);
                    // Back-to-back reasoning: only the one adjacent to the
                    // next output item can be paired with it.
                    if matches!(items.last(), Some(InputItem::Reasoning { .. })) {
                        items.pop();
                    }
                    items.push(reasoning);
                }
            }
            _ => {}
        }
    }
    flush_text(&mut text, &mut items);
    // Trailing reasoning has nothing to precede.
    while matches!(items.last(), Some(InputItem::Reasoning { .. })) {
        items.pop();
    }
    out.extend(items);
    Ok(())
}

/// A `reasoning` input item for a `Thinking` block holding this provider's
/// encrypted reasoning; `None` for anything else.
fn replayable_reasoning(block: &ContentBlock) -> Option<InputItem> {
    let ContentBlock::Thinking {
        text,
        signature: Some(signature),
    } = block
    else {
        return None;
    };
    let encrypted_content = signature.strip_prefix(REASONING_SIGNATURE_PREFIX)?;
    let summary = if text.is_empty() {
        Vec::new()
    } else {
        vec![InputSummaryPart::SummaryText { text: text.clone() }]
    };
    Some(InputItem::Reasoning {
        summary,
        encrypted_content: encrypted_content.to_string(),
    })
}

fn push_tool_output_items(msg: &Message, out: &mut Vec<InputItem>, supports_vision: bool) {
    let mut deferred_images: Vec<(String, Vec<InputContentPart>)> = Vec::new();
    for block in &msg.content {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error: _,
        } = block
        {
            let (output, images) =
                split_images(content, ImagePolicy::openai(supports_vision), input_image);
            if !images.is_empty() {
                deferred_images.push((tool_use_id.as_str().to_string(), images));
            }
            out.push(InputItem::FunctionCallOutput {
                call_id: tool_use_id.as_str().to_string(),
                output,
            });
        }
    }
    if let Some(content) =
        tool_images_message_parts(deferred_images, |text| InputContentPart::InputText { text })
    {
        out.push(InputItem::Message {
            role: "user".into(),
            content,
        });
    }
}

pub fn parse_responses_response(resp: ResponsesResponse) -> LlmResult<MessageResponse> {
    if let Some(err) = resp.error {
        return Err(LlmError::Provider {
            code: "response_failed".into(),
            message: err.to_string(),
        });
    }

    let mut content: Vec<ContentBlock> = Vec::new();
    for item in resp.output {
        match item {
            OutputItem::Message { content: parts, .. } => {
                for part in parts {
                    match part {
                        OutputContentPart::OutputText { text } => {
                            content.push(ContentBlock::text(text));
                        }
                        OutputContentPart::Refusal { refusal } => {
                            content.push(ContentBlock::text(format!("[refused] {refusal}")));
                        }
                        OutputContentPart::Unknown => {}
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let raw = arguments.unwrap_or_default();
                let input: serde_json::Value = if raw.trim().is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&raw).map_err(|e| {
                        LlmError::MalformedResponse(format!(
                            "function_call.arguments is not valid JSON: {e}"
                        ))
                    })?
                };
                // An empty id or name would break the engine's
                // tool_use/tool_result pairing invariant and get rejected
                // by providers on replay — fail loudly like the Anthropic
                // converter does for malformed blocks.
                let call_id = match call_id {
                    Some(id) if !id.is_empty() => id,
                    _ => {
                        return Err(LlmError::MalformedResponse(
                            "function_call item missing call_id".into(),
                        ))
                    }
                };
                let name = match name {
                    Some(name) if !name.is_empty() => name,
                    _ => {
                        return Err(LlmError::MalformedResponse(
                            "function_call item missing name".into(),
                        ))
                    }
                };
                content.push(ContentBlock::ToolUse {
                    id: ToolUseId::new(call_id),
                    name,
                    input,
                });
            }
            // Surface the summary as Thinking, and keep the encrypted
            // reasoning on it so the next request can replay it.
            OutputItem::Reasoning {
                summary,
                encrypted_content,
                ..
            } => {
                let text = summary
                    .into_iter()
                    .filter_map(|part| match part {
                        OutputSummaryPart::SummaryText { text } => Some(text),
                        OutputSummaryPart::Unknown => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let signature = encrypted_content
                    .filter(|c| !c.is_empty())
                    .map(|c| format!("{REASONING_SIGNATURE_PREFIX}{c}"));
                if signature.is_some() || !text.is_empty() {
                    content.push(ContentBlock::Thinking { text, signature });
                }
            }
            OutputItem::Unknown => {}
        }
    }

    let stop_reason = stop_reason_for(
        resp.status.as_deref(),
        resp.incomplete_details
            .as_ref()
            .and_then(|d| d.reason.clone()),
        content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. })),
    );

    Ok(MessageResponse {
        id: resp.id,
        message: Message {
            id: MessageId::new(),
            role: Role::Assistant,
            content,
            created_at: chrono::DateTime::<chrono::Utc>::from(SystemTime::now()),
        },
        usage: resp.usage.map(Into::into).unwrap_or_default(),
        stop_reason,
    })
}

/// Stop-reason derivation shared with the SSE state machine (`sse.rs`
/// `StreamState::finish`), so streaming and non-streaming responses
/// classify identically.
pub(super) fn stop_reason_for(
    status: Option<&str>,
    incomplete_reason: Option<String>,
    has_tool_call: bool,
) -> StopReason {
    if status == Some("incomplete") {
        return match incomplete_reason.as_deref() {
            Some("max_output_tokens") => StopReason::MaxTokens,
            Some(other) => StopReason::Other(other.to_string()),
            None => StopReason::MaxTokens,
        };
    }
    if has_tool_call {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use snaca_core::ENGINE_FEEDBACK_PREFIX;

    fn user(text: &str) -> Message {
        Message::user_text(text)
    }

    #[test]
    fn system_prompt_becomes_instructions() {
        // Without this: a request with no max_tokens could go out uncapped (a
        // reasoning model then runs to its model maximum), and the system
        // prompt or `store: false` could miss their top-level fields.
        let req = MessageRequest::new("gpt-test")
            .with_system("You are SNACA")
            .with_messages(vec![user("hi")]);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.instructions.as_deref(), Some("You are SNACA"));
        assert_eq!(wire.input.len(), 1);
        assert!(!wire.store);
        assert_eq!(wire.max_output_tokens, Some(DEFAULT_MAX_OUTPUT_TOKENS));
    }

    #[test]
    fn history_system_messages_join_instructions() {
        // Without this: a Role::System message in history would be dropped
        // (or misordered) instead of joining `instructions`, so an SDK
        // caller's extra system guidance never reaches a Responses model.
        let req = MessageRequest::new("gpt-test")
            .with_system("Top-level")
            .with_messages(vec![Message::system_text("From history"), user("hi")]);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(
            wire.instructions.as_deref(),
            Some("Top-level\n\nFrom history")
        );
    }

    #[test]
    fn assistant_replays_output_text_and_function_calls() {
        // Without this: assistant text could replay as `input_text` (or call
        // arguments lose their JSON-string encoding) and the API would reject
        // every follow-up request in the tool loop.
        let assistant = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::thinking("hidden reasoning"),
                ContentBlock::text("let me check"),
                ContentBlock::tool_use("call_1", "Read", json!({"path": "x"})),
            ],
        );
        let req = MessageRequest::new("gpt-test").with_messages(vec![assistant]);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.input.len(), 2, "thinking block is not replayed");
        match &wire.input[0] {
            InputItem::Message { role, content } => {
                assert_eq!(role, "assistant");
                assert!(matches!(
                    &content[0],
                    InputContentPart::OutputText { text } if text == "let me check"
                ));
            }
            other => panic!("expected Message item, got {other:?}"),
        }
        match &wire.input[1] {
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "Read");
                assert_eq!(arguments, "{\"path\":\"x\"}");
            }
            other => panic!("expected FunctionCall item, got {other:?}"),
        }
    }

    fn signed_thinking(text: &str, encrypted: &str) -> ContentBlock {
        ContentBlock::Thinking {
            text: text.into(),
            signature: Some(format!("{REASONING_SIGNATURE_PREFIX}{encrypted}")),
        }
    }

    fn tool_result(id: &str) -> Message {
        Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                snaca_core::ToolUseId::new(id),
                vec![ContentBlock::text("ok")],
            )],
        )
    }

    fn input_json(req: &MessageRequest) -> Vec<serde_json::Value> {
        let wire = build_responses_request(req, false, true).unwrap();
        serde_json::to_value(&wire).unwrap()["input"]
            .as_array()
            .unwrap()
            .clone()
    }

    #[test]
    fn replays_reasoning_without_id_right_before_its_function_call() {
        // Without this: GPT-5 / o-series lose their reasoning between tool
        // calls (OpenAI asks for it to be replayed). Sending an `id` under
        // `store: false` 400s ("Item with id 'rs_…' not found"), and a
        // reasoning item not directly followed by its output item 400s
        // ("provided without its required following item").
        let req = MessageRequest::new("gpt-test").with_messages(vec![
            user("read x"),
            Message::new(
                Role::Assistant,
                vec![
                    signed_thinking("plan", "gAAA"),
                    ContentBlock::tool_use("call_1", "Read", json!({"path": "x"})),
                ],
            ),
            tool_result("call_1"),
        ]);
        let input = input_json(&req);
        assert_eq!(
            input[1],
            json!({"type": "reasoning",
                   "summary": [{"type": "summary_text", "text": "plan"}],
                   "encrypted_content": "gAAA"}),
            "id-less, exact shape"
        );
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
    }

    #[test]
    fn max_output_tokens_is_raised_to_the_api_minimum() {
        // Without this: the engine's 1-token content-filter probe is sent
        // as `max_output_tokens: 1`, which the API rejects (minimum 16), so
        // a flagged history message is never located and redacted and the
        // thread stays stuck.
        let req = MessageRequest::new("gpt-test")
            .with_messages(vec![user("hi")])
            .with_max_tokens(1);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.max_output_tokens, Some(16));
        let req = req.with_max_tokens(2048);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.max_output_tokens, Some(2048));
    }

    #[test]
    fn reasoning_effort_is_sent_only_when_configured() {
        // Without this: a configured effort never reaches the API (some
        // deployments then barely reason once tools are offered, so there is
        // no reasoning to keep), or a default one is forced on endpoints
        // that should pick their own.
        let req = MessageRequest::new("gpt-test").with_messages(vec![user("hi")]);
        let with = build_responses_request_with(
            &req,
            false,
            true,
            ReasoningMode {
                effort: Some("medium".into()),
                ..ReasoningMode::default()
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&with).unwrap()["reasoning"],
            json!({"effort": "medium"})
        );
        let without = build_responses_request(&req, false, true).unwrap();
        assert!(serde_json::to_value(&without)
            .unwrap()
            .get("reasoning")
            .is_none());
    }

    #[test]
    fn reasoning_before_the_last_user_message_is_not_replayed() {
        // Without this: every finished turn's encrypted reasoning (several
        // KB each) would be re-sent on every later request — payload grows
        // with thread length for reasoning the model ignores anyway.
        let req = MessageRequest::new("gpt-test").with_messages(vec![
            user("q1"),
            Message::new(
                Role::Assistant,
                vec![signed_thinking("plan", "gAAA"), ContentBlock::text("a1")],
            ),
            user("q2"),
        ]);
        let input = input_json(&req);
        assert!(
            input.iter().all(|item| item["type"] != "reasoning"),
            "{input:#?}"
        );
    }

    #[test]
    fn engine_feedback_mid_loop_keeps_the_reasoning_before_it() {
        // Without this: retry feedback the engine injects as a user message
        // would end the replay window, so the loop's earlier function call
        // would go out without the reasoning that produced it.
        let req = MessageRequest::new("gpt-test").with_messages(vec![
            user("read x"),
            Message::new(
                Role::Assistant,
                vec![
                    signed_thinking("plan", "gAAA"),
                    ContentBlock::tool_use("call_1", "Read", json!({"path": "x"})),
                ],
            ),
            tool_result("call_1"),
            user(&format!("{ENGINE_FEEDBACK_PREFIX}that failed twice")),
        ]);
        let input = input_json(&req);
        assert_eq!(input[1]["type"], "reasoning", "{input:#?}");
        assert_eq!(input[2]["call_id"], "call_1");
    }

    #[test]
    fn foreign_thinking_signature_is_not_replayed() {
        // Without this: in a thread that switched from Anthropic, its
        // thinking signature would be sent as `encrypted_content` and the
        // API would reject the request as unverifiable.
        let req = MessageRequest::new("gpt-test").with_messages(vec![
            user("q"),
            Message::new(
                Role::Assistant,
                vec![
                    ContentBlock::Thinking {
                        text: "plan".into(),
                        signature: Some("EqQBCkgIARABGAIi-anthropic".into()),
                    },
                    ContentBlock::tool_use("call_1", "Read", json!({})),
                ],
            ),
            tool_result("call_1"),
        ]);
        let input = input_json(&req);
        assert!(input.iter().all(|item| item["type"] != "reasoning"));
    }

    #[test]
    fn unpaired_reasoning_is_trimmed() {
        // Without this: a reasoning item followed by another reasoning item
        // or by nothing gets the request rejected ("provided without its
        // required following item"), wedging the whole tool loop.
        let req = MessageRequest::new("gpt-test").with_messages(vec![
            user("q"),
            Message::new(
                Role::Assistant,
                vec![
                    signed_thinking("", "first"),
                    signed_thinking("", "second"),
                    ContentBlock::tool_use("call_1", "Read", json!({})),
                    signed_thinking("", "trailing"),
                ],
            ),
            tool_result("call_1"),
        ]);
        let input = input_json(&req);
        let kinds: Vec<&str> = input.iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            [
                "message",
                "reasoning",
                "function_call",
                "function_call_output"
            ]
        );
        assert_eq!(input[1]["encrypted_content"], "second");
        assert_eq!(input[1]["summary"], json!([]), "empty summary is []");
    }

    #[test]
    fn animated_gif_in_history_is_not_sent() {
        // Without this: see the deepseek converter's test — the Responses
        // wire also accepts only non-animated GIFs.
        let req = MessageRequest::new("gpt-test").with_messages(vec![Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                snaca_core::ToolUseId::new("call_1"),
                vec![ContentBlock::Image {
                    source: snaca_core::ImageSource::Base64 {
                        media_type: "image/gif".into(),
                        data: crate::images::tests::gif_base64(2),
                    },
                }],
            )],
        )]);
        let wire =
            serde_json::to_string(&build_responses_request(&req, false, true).unwrap()).unwrap();
        assert!(!wire.contains("input_image"), "{wire}");
        assert!(wire.contains("animated GIF"), "{wire}");
    }

    #[test]
    fn vision_on_tool_result_image_deferred_to_user_message() {
        // Without this: an image a tool returned (a Read of a screenshot)
        // never reaches the model, which answers from the filename alone.
        let tool_msg = Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new("call_1"),
                vec![
                    ContentBlock::text("cat.png:"),
                    ContentBlock::Image {
                        source: snaca_core::ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "iVBORw0KGgoAAEC".into(),
                        },
                    },
                ],
            )],
        );
        let req = MessageRequest::new("gpt-test").with_messages(vec![tool_msg]);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.input.len(), 2);
        assert!(
            matches!(&wire.input[0], InputItem::FunctionCallOutput { call_id, .. } if call_id == "call_1")
        );
        match &wire.input[1] {
            InputItem::Message { role, content } => {
                assert_eq!(role, "user");
                assert!(matches!(&content[0], InputContentPart::InputText { text }
                    if text.contains("call_1")));
                assert!(matches!(
                    &content[1],
                    InputContentPart::InputImage { image_url, .. }
                        if image_url == "data:image/png;base64,iVBORw0KGgoAAEC"
                ));
            }
            other => panic!("expected user Message item, got {other:?}"),
        }
    }

    #[test]
    fn tools_serialize_flat_with_strict_false() {
        // Without this: tools could go out in the nested Chat Completions
        // shape or with `strict: true`, which SNACA's hand-built schemas
        // don't satisfy, and every request that offers tools would 400.
        let req = MessageRequest::new("gpt-test")
            .with_messages(vec![user("hi")])
            .with_tools(vec![crate::request::ToolSchema {
                name: "Read".into(),
                description: "read".into(),
                input_schema: json!({"type": "object"}),
            }]);
        let wire = build_responses_request(&req, false, true).unwrap();
        assert_eq!(wire.tools.len(), 1);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["tools"][0]["type"], json!("function"));
        assert_eq!(json["tools"][0]["name"], json!("Read"));
        assert_eq!(json["tools"][0]["strict"], json!(false));
        assert_eq!(json["tools"][0]["parameters"]["type"], json!("object"));
    }

    fn make_response(json: serde_json::Value) -> ResponsesResponse {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn parse_function_call_response() {
        // Without this: a mis-parsed call_id or arguments would run the tool
        // with the wrong input or leave its result unpaired, and an empty
        // reasoning item would leave an empty Thinking block in history.
        let resp = make_response(json!({
            "id": "resp_2",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{\"path\":\"x\"}"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7}
        }));
        let parsed = parse_responses_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        // A reasoning item with neither summary nor encrypted content
        // leaves nothing to keep; only the function call surfaces.
        assert_eq!(parsed.message.content.len(), 1);
        match &parsed.message.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id.as_str(), "call_1");
                assert_eq!(name, "Read");
                assert_eq!(input, &json!({"path": "x"}));
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parse_function_call_missing_call_id_errors() {
        // Without this: a function_call with no call_id would become a
        // ToolUse with an empty id, so the engine can't pair its tool_result
        // and every provider rejects the replayed history.
        let resp = make_response(json!({
            "id": "resp_bad",
            "status": "completed",
            "output": [{"type": "function_call", "name": "Read", "arguments": "{}"}]
        }));
        let err = parse_responses_response(resp).unwrap_err();
        assert!(matches!(err, LlmError::MalformedResponse(msg) if msg.contains("call_id")));
    }

    #[test]
    fn parse_incomplete_maps_to_max_tokens() {
        // Without this: a truncated non-streamed reply is reported as
        // EndTurn, so the engine's max-tokens escalation never fires and the
        // answer stops mid-sentence.
        let resp = make_response(json!({
            "id": "resp_3",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}]}]
        }));
        let parsed = parse_responses_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn parse_refusal_surfaces_as_text() {
        // Without this: a refusal would end the turn as an empty assistant
        // reply, and the user would never learn the model refused.
        let resp = make_response(json!({
            "id": "resp_4",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "cannot help with that"}]
            }]
        }));
        let parsed = parse_responses_response(resp).unwrap();
        match &parsed.message.content[0] {
            ContentBlock::Text { text } => assert!(text.contains("cannot help")),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn parse_reasoning_keeps_summary_and_encrypted_content() {
        // Without this: the encrypted reasoning returned by a non-streaming
        // call is dropped at parse time and there is nothing to replay.
        let resp = make_response(json!({
            "id": "resp_r",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA",
                 "summary": [{"type": "summary_text", "text": "check the file"}]},
                {"type": "function_call", "call_id": "call_1", "name": "Read",
                 "arguments": "{}"}
            ]
        }));
        let parsed = parse_responses_response(resp).unwrap();
        assert_eq!(
            parsed.message.content[0],
            ContentBlock::Thinking {
                text: "check the file".into(),
                signature: Some("openai-responses:gAAA".into()),
            }
        );
        assert!(matches!(
            &parsed.message.content[1],
            ContentBlock::ToolUse { .. }
        ));
    }

    #[test]
    fn parse_unknown_output_items_are_skipped() {
        // Without this: one output item type we don't model (web_search_call,
        // a new kind, a gateway extra) fails deserialisation of the whole
        // reply, which is lost as MalformedResponse.
        let resp = make_response(json!({
            "id": "resp_6",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1"},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "done"}]}
            ]
        }));
        let parsed = parse_responses_response(resp).unwrap();
        assert_eq!(parsed.message.content.len(), 1);
    }
}

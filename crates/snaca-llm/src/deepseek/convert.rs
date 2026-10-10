//! Canonical ↔ DeepSeek wire conversion.
//!
//! Two directions:
//! - **Outbound (request)**: `MessageRequest` → `ChatRequest`. System prompt
//!   becomes a `role: "system"` message. `Role::Tool` messages are split:
//!   each `ToolResult` block becomes its own `role: "tool"` wire message
//!   (OpenAI convention). `Thinking` text is replayed as the assistant
//!   message's `reasoning_content` (DeepSeek's thinking models require it);
//!   signatures are never sent. `Image` blocks follow the OpenAI multimodal
//!   conventions: user messages carry `image_url` content parts (data URLs
//!   for base64) when `supports_vision` is on; tool-result images are
//!   text-only on the wire, so they are re-surfaced as a follow-up user
//!   message. With vision off (the default — DeepSeek's hosted models are
//!   text-only), images degrade to `[image omitted: …]` placeholder text.
//! - **Inbound (response)**: `ChatResponse` → `MessageResponse`. Multiple
//!   `tool_calls` become multiple `ContentBlock::ToolUse` blocks; the order
//!   is preserved. `reasoning_content` (R1) becomes a `ContentBlock::Thinking`
//!   inserted before the text block.

use crate::deepseek::wire::{
    ChatRequest, ChatResponse, WireContent, WireContentPart, WireImageUrl, WireMessage, WireTool,
    WireToolCall, WireToolCallFunction, WireToolDefinition,
};
use crate::error::{LlmError, LlmResult};
use crate::images::{image_url_of, split_images, tool_images_message_parts, ImagePolicy};
use crate::request::{MessageRequest, ToolSchema};
use crate::response::{MessageResponse, StopReason};
use snaca_core::{ContentBlock, ImageSource, Message, MessageId, Role, ToolUseId};
use std::time::SystemTime;

/// Test shorthand for the text-only default (DeepSeek's hosted
/// `deepseek-chat` / `deepseek-reasoner` models are text-only). The
/// client always calls [`build_chat_request_with_vision`].
#[cfg(test)]
pub fn build_chat_request(req: &MessageRequest, stream: bool) -> LlmResult<ChatRequest> {
    build_chat_request_with_vision(req, stream, false)
}

pub fn build_chat_request_with_vision(
    req: &MessageRequest,
    stream: bool,
    supports_vision: bool,
) -> LlmResult<ChatRequest> {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);

    // Top-level system prompt → first system wire message. DeepSeek's
    // context cache is transparent and has no `cache_control` knob, so
    // the segmentation is irrelevant on the wire — flatten back to a
    // single string and let the platform deduplicate prefixes itself.
    if let Some(system) = req.flat_system() {
        if !system.is_empty() {
            messages.push(WireMessage {
                role: "system".into(),
                content: Some(WireContent::Text(system)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    for msg in &req.messages {
        push_message(msg, &mut messages, supports_vision)?;
    }

    let tools = req.tools.iter().map(tool_schema_to_wire).collect();

    Ok(ChatRequest {
        model: req.model.clone(),
        messages,
        tools,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        stop: if req.stop_sequences.is_empty() {
            None
        } else {
            Some(req.stop_sequences.clone())
        },
        stream,
    })
}

fn push_message(msg: &Message, out: &mut Vec<WireMessage>, supports_vision: bool) -> LlmResult<()> {
    match msg.role {
        Role::System => {
            let text = ContentBlock::collect_text(&msg.content);
            out.push(WireMessage {
                role: "system".into(),
                content: Some(WireContent::Text(text)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        Role::User => {
            // Images the gate refuses (text-only model, unsupported
            // format, oversized) degrade to explicit placeholder lines so
            // the model at least knows an image was there.
            let (text, images) = split_images(
                &msg.content,
                ImagePolicy::openai(supports_vision),
                image_url_part,
            );
            let content = if images.is_empty() {
                WireContent::Text(text)
            } else {
                // OpenAI-style multimodal user message: text part first
                // (when non-empty), then image parts in order.
                let mut parts = Vec::with_capacity(images.len() + 1);
                if !text.is_empty() {
                    parts.push(WireContentPart::Text { text });
                }
                parts.extend(images);
                WireContent::Parts(parts)
            };
            out.push(WireMessage {
                role: "user".into(),
                content: Some(content),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        Role::Assistant => {
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut tool_calls: Vec<WireToolCall> = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text: t } => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                    ContentBlock::Thinking { text: t, .. } => {
                        // DeepSeek thinking models (V3.1+, V4) require the
                        // prior turn's reasoning_content to be replayed in
                        // history — omitting it triggers
                        // `invalid_request_error: The reasoning_content in
                        // the thinking mode must be passed back to the API`.
                        // Non-thinking models ignore the extra field.
                        if !reasoning.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(t);
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_calls.push(WireToolCall {
                            id: id.as_str().to_string(),
                            call_type: "function".into(),
                            function: WireToolCallFunction {
                                name: name.clone(),
                                arguments: serde_json::to_string(input)?,
                            },
                        });
                    }
                    // Assistant should not contain ToolResult or Image (in M1).
                    _ => {}
                }
            }
            // Drop an assistant turn that yielded nothing — DeepSeek
            // rejects an assistant message with neither `content` nor
            // `tool_calls` (`Invalid assistant message: content or
            // tool_calls must be set`). The engine no longer persists
            // empty responses, but stale rows from before that fix can
            // still appear in history; skipping them keeps the thread
            // replayable. Matches the Anthropic converter's behavior.
            if text.is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
                return Ok(());
            }
            out.push(WireMessage {
                role: "assistant".into(),
                content: if text.is_empty() {
                    None
                } else {
                    Some(WireContent::Text(text))
                },
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
                name: None,
            });
        }
        Role::Tool => {
            // Each ToolResult becomes its own `role: "tool"` wire message.
            // OpenAI tool messages are text-only, so any image blocks a
            // tool returned are deferred: with vision they are re-surfaced
            // as one follow-up user message (the standard OpenAI-family
            // workaround); without vision they degrade to placeholder text
            // appended to the tool message itself.
            let mut deferred_images: Vec<(String, Vec<WireContentPart>)> = Vec::new();
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error: _,
                } = block
                {
                    let (text, images) = split_images(
                        content,
                        ImagePolicy::openai(supports_vision),
                        image_url_part,
                    );
                    if !images.is_empty() {
                        deferred_images.push((tool_use_id.as_str().to_string(), images));
                    }
                    out.push(WireMessage {
                        role: "tool".into(),
                        content: Some(WireContent::Text(text)),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id.as_str().to_string()),
                        name: None,
                    });
                }
            }
            if let Some(parts) =
                tool_images_message_parts(deferred_images, |text| WireContentPart::Text { text })
            {
                out.push(WireMessage {
                    role: "user".into(),
                    content: Some(WireContent::Parts(parts)),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
        }
    }
    Ok(())
}

fn image_url_part(source: &ImageSource) -> WireContentPart {
    WireContentPart::ImageUrl {
        image_url: WireImageUrl {
            url: image_url_of(source),
        },
    }
}

fn tool_schema_to_wire(s: &ToolSchema) -> WireTool {
    WireTool {
        tool_type: "function".into(),
        function: WireToolDefinition {
            name: s.name.clone(),
            description: s.description.clone(),
            parameters: s.input_schema.clone(),
        },
    }
}

pub fn parse_chat_response(resp: ChatResponse) -> LlmResult<MessageResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::MalformedResponse("no choices in response".into()))?;

    let mut content: Vec<ContentBlock> = Vec::new();

    if let Some(reasoning) = choice
        .message
        .reasoning_content
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        content.push(ContentBlock::Thinking {
            text: reasoning.to_string(),
            signature: None,
        });
    }

    if let Some(text) = choice.message.content.as_deref().filter(|s| !s.is_empty()) {
        content.push(ContentBlock::text(text));
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = if tc.function.arguments.is_empty() {
                serde_json::Value::Object(Default::default())
            } else {
                serde_json::from_str(&tc.function.arguments).map_err(|e| {
                    LlmError::MalformedResponse(format!(
                        "tool_call.arguments is not valid JSON: {e}"
                    ))
                })?
            };
            content.push(ContentBlock::ToolUse {
                id: ToolUseId::new(tc.id),
                name: tc.function.name,
                input,
            });
        }
    }

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("stop_sequence") => StopReason::StopSequence,
        Some(other) => StopReason::Other(other.to_string()),
        // No finish_reason: pick something safe — if there's a tool call, it's
        // ToolUse, otherwise EndTurn.
        None => {
            if content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }
        }
    };

    let usage = resp.usage.map(Into::into).unwrap_or_default();

    Ok(MessageResponse {
        id: resp.id,
        message: Message {
            id: MessageId::new(),
            role: Role::Assistant,
            content,
            // We don't have a server timestamp; use receive time.
            created_at: chrono::DateTime::<chrono::Utc>::from(SystemTime::now()),
        },
        usage,
        stop_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use snaca_core::{ContentBlock, Message, Role};

    #[test]
    fn animated_gif_in_history_is_not_sent() {
        // Without this: a tool result from before the upgrade holding an
        // animated GIF is replayed as an image part on every request, and an
        // OpenAI-family endpoint rejects the whole thread.
        let tool = Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new("call_1"),
                vec![ContentBlock::Image {
                    source: snaca_core::ImageSource::Base64 {
                        media_type: "image/gif".into(),
                        data: crate::images::tests::gif_base64(2),
                    },
                }],
            )],
        );
        let req = MessageRequest::new("m").with_messages(vec![tool]);
        let wire =
            serde_json::to_string(&build_chat_request_with_vision(&req, false, true).unwrap())
                .unwrap();
        assert!(!wire.contains("image_url"), "{wire}");
        assert!(wire.contains("animated GIF"), "{wire}");
    }

    fn user(text: &str) -> Message {
        Message::user_text(text)
    }

    fn assistant_text(text: &str) -> Message {
        Message::assistant_text(text)
    }

    fn assistant_tool_call(id: &str, name: &str, args: serde_json::Value) -> Message {
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_use(id, name, args)],
        )
    }

    fn tool_result(id: &str, text: &str) -> Message {
        Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new(id),
                vec![ContentBlock::text(text)],
            )],
        )
    }

    fn tool_schema(name: &str) -> ToolSchema {
        ToolSchema {
            name: name.to_string(),
            description: format!("description of {name}"),
            input_schema: json!({"type": "object"}),
        }
    }

    /// Flat string view of a wire message's content, for assertions on
    /// messages that are still emitted in the plain-string shape.
    fn content_text(msg: &WireMessage) -> Option<&str> {
        msg.content.as_ref().and_then(|c| c.as_text())
    }

    #[test]
    fn system_prompt_becomes_first_message() {
        let req = MessageRequest::new("deepseek-chat")
            .with_system("You are SNACA")
            .with_messages(vec![user("hi")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(wire.messages.len(), 2);
        assert_eq!(wire.messages[0].role, "system");
        assert_eq!(content_text(&wire.messages[0]), Some("You are SNACA"));
        assert_eq!(wire.messages[1].role, "user");
    }

    #[test]
    fn segmented_system_flattens_back_into_one_system_message() {
        use crate::request::SystemSegment;
        let req = MessageRequest::new("deepseek-chat")
            .with_system_segments(vec![
                SystemSegment::cacheable("BASE"),
                SystemSegment::cacheable("MEMORY"),
                SystemSegment::volatile("RECALL"),
            ])
            .with_messages(vec![user("hi")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(wire.messages[0].role, "system");
        assert_eq!(
            content_text(&wire.messages[0]),
            Some("BASE\n\nMEMORY\n\nRECALL")
        );
    }

    #[test]
    fn assistant_text_only_serializes_content() {
        let req = MessageRequest::new("deepseek-chat").with_messages(vec![assistant_text("ok")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(content_text(&wire.messages[0]), Some("ok"));
        assert!(wire.messages[0].tool_calls.is_none());
    }

    #[test]
    fn assistant_tool_call_serializes_tool_calls() {
        let req = MessageRequest::new("deepseek-chat").with_messages(vec![assistant_tool_call(
            "call_1",
            "Read",
            json!({"path": "x"}),
        )]);
        let wire = build_chat_request(&req, false).unwrap();
        let tcs = wire.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].function.name, "Read");
        assert_eq!(tcs[0].function.arguments, "{\"path\":\"x\"}");
        // Content should be omitted when empty.
        assert!(wire.messages[0].content.is_none());
    }

    #[test]
    fn tool_result_split_into_role_tool_messages() {
        let req = MessageRequest::new("deepseek-chat")
            .with_messages(vec![tool_result("call_1", "file contents")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(wire.messages.len(), 1);
        assert_eq!(wire.messages[0].role, "tool");
        assert_eq!(wire.messages[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(content_text(&wire.messages[0]), Some("file contents"));
    }

    #[test]
    fn empty_assistant_message_is_skipped() {
        // Stale empty assistant rows from before the engine-side guard
        // must not be forwarded — DeepSeek would reject the request
        // with `Invalid assistant message: content or tool_calls must
        // be set` on every subsequent turn.
        let empty_assistant = Message::new(Role::Assistant, vec![]);
        let req = MessageRequest::new("deepseek-chat")
            .with_messages(vec![empty_assistant, assistant_text("ok")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(wire.messages.len(), 1);
        assert_eq!(content_text(&wire.messages[0]), Some("ok"));
    }

    #[test]
    fn thinking_blocks_replayed_as_reasoning_content() {
        // DeepSeek's V3.1+/V4 thinking models require the prior
        // `reasoning_content` to be echoed in history; older non-thinking
        // models tolerate the extra field, so we always emit it.
        let assistant_with_thinking = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::thinking("think think"),
                ContentBlock::text("answer"),
            ],
        );
        let req = MessageRequest::new("deepseek-chat").with_messages(vec![assistant_with_thinking]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(content_text(&wire.messages[0]), Some("answer"));
        assert_eq!(
            wire.messages[0].reasoning_content.as_deref(),
            Some("think think")
        );
    }

    #[test]
    fn tools_serialize_as_function_type() {
        let req = MessageRequest::new("deepseek-chat").with_tools(vec![tool_schema("Read")]);
        let wire = build_chat_request(&req, false).unwrap();
        assert_eq!(wire.tools.len(), 1);
        assert_eq!(wire.tools[0].tool_type, "function");
        assert_eq!(wire.tools[0].function.name, "Read");
    }

    fn make_chat_response(json: serde_json::Value) -> ChatResponse {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn parse_text_response() {
        let resp = make_chat_response(json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::EndTurn);
        assert_eq!(parsed.usage.input_tokens, 5);
        assert_eq!(parsed.usage.output_tokens, 1);
        match &parsed.message.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn parse_tool_call_response() {
        let resp = make_chat_response(json!({
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": "{\"path\":\"x\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        assert_eq!(parsed.message.content.len(), 1);
        match &parsed.message.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id.as_str(), "call_1");
                assert_eq!(name, "Read");
                assert_eq!(input, &json!({"path": "x"}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_with_reasoning_content() {
        let resp = make_chat_response(json!({
            "id": "chatcmpl-3",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reasoning trace"
                },
                "finish_reason": "stop"
            }]
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.message.content.len(), 2);
        assert!(matches!(
            &parsed.message.content[0],
            ContentBlock::Thinking { text, .. } if text == "reasoning trace"
        ));
        assert!(matches!(
            &parsed.message.content[1],
            ContentBlock::Text { text } if text == "answer"
        ));
    }

    #[test]
    fn parse_response_handles_missing_finish_reason() {
        // Falls back to EndTurn when there are no tool calls and no reason.
        let resp = make_chat_response(json!({
            "id": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn parse_response_falls_back_to_tool_use_on_missing_reason_with_tool_calls() {
        let resp = make_chat_response(json!({
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "X", "arguments": "{}"}
                    }]
                }
            }]
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn parse_invalid_tool_arguments_errors() {
        let resp = make_chat_response(json!({
            "id": "x",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "c",
                        "type": "function",
                        "function": {"name": "X", "arguments": "not json {{"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        let err = parse_chat_response(resp).unwrap_err();
        assert!(matches!(err, LlmError::MalformedResponse(_)));
    }

    #[test]
    fn empty_choices_errors() {
        let resp = make_chat_response(json!({"id": "x", "choices": []}));
        let err = parse_chat_response(resp).unwrap_err();
        assert!(matches!(err, LlmError::MalformedResponse(_)));
    }

    #[test]
    fn cache_tokens_propagate_to_usage() {
        let resp = make_chat_response(json!({
            "id": "x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20
            }
        }));
        let parsed = parse_chat_response(resp).unwrap();
        assert_eq!(parsed.usage.cache_read_input_tokens, Some(80));
        assert_eq!(parsed.usage.cache_creation_input_tokens, Some(20));
    }

    fn user_msg_with_image() -> Message {
        Message::new(
            Role::User,
            vec![
                ContentBlock::text("what is in this picture?"),
                ContentBlock::Image {
                    source: snaca_core::ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "iVBORw0KGgoAAEC".into(),
                    },
                },
            ],
        )
    }

    #[test]
    fn vision_on_user_image_becomes_content_parts() {
        // Without this: an image a user sends to a vision model over Chat
        // Completions would go out as text only, or with a data URL the
        // endpoint can't parse, and the model never sees it.
        let req = MessageRequest::new("vision-model").with_messages(vec![user_msg_with_image()]);
        let wire = build_chat_request_with_vision(&req, false, true).unwrap();
        let msg = &wire.messages[0];
        assert_eq!(msg.role, "user");
        match &msg.content {
            Some(WireContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert!(
                    matches!(&parts[0], WireContentPart::Text { text } if text == "what is in this picture?")
                );
                match &parts[1] {
                    WireContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "data:image/png;base64,iVBORw0KGgoAAEC");
                    }
                    other => panic!("expected ImageUrl part, got {other:?}"),
                }
            }
            other => panic!("expected Parts content, got {other:?}"),
        }
        // Serialises to OpenAI's multimodal shape.
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(
            json["messages"][0]["content"][1]["type"],
            json!("image_url")
        );
        assert_eq!(
            json["messages"][0]["content"][1]["image_url"]["url"],
            json!("data:image/png;base64,iVBORw0KGgoAAEC")
        );
    }

    #[test]
    fn vision_on_url_image_passes_url_through() {
        // Without this: a URL image from a custom tool or SDK caller could be
        // mangled on the Chat Completions wire, or ride behind an empty text
        // part that stricter OpenAI-compatible endpoints reject.
        let req = MessageRequest::new("vision-model").with_messages(vec![Message::new(
            Role::User,
            vec![ContentBlock::Image {
                source: snaca_core::ImageSource::Url {
                    url: "https://example.com/cat.png".into(),
                },
            }],
        )]);
        let wire = build_chat_request_with_vision(&req, false, true).unwrap();
        match &wire.messages[0].content {
            Some(WireContent::Parts(parts)) => {
                assert_eq!(parts.len(), 1, "empty text part is dropped");
                assert!(matches!(
                    &parts[0],
                    WireContentPart::ImageUrl { image_url }
                        if image_url.url == "https://example.com/cat.png"
                ));
            }
            other => panic!("expected Parts content, got {other:?}"),
        }
    }

    #[test]
    fn vision_off_user_image_degrades_to_placeholder_text() {
        // Without this: on DeepSeek's text-only models (the default) a user's
        // image goes out as an image_url part and the request is rejected — on
        // every later turn too, since the image stays in history.
        let req = MessageRequest::new("deepseek-chat").with_messages(vec![user_msg_with_image()]);
        let wire = build_chat_request_with_vision(&req, false, false).unwrap();
        assert_eq!(wire.messages[0].role, "user");
        match &wire.messages[0].content {
            Some(WireContent::Text(text)) => {
                assert!(text.contains("what is in this picture?"));
                assert!(text.contains("[image omitted: image/png"));
            }
            other => panic!("expected Text content, got {other:?}"),
        }
    }

    #[test]
    fn vision_on_tool_result_image_deferred_to_user_message() {
        // Without this: Chat Completions tool messages are text-only, so an
        // image a tool returns would be dropped and the vision model answers
        // without ever seeing it.
        let tool_result = Message::new(
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
        let req = MessageRequest::new("vision-model").with_messages(vec![tool_result]);
        let wire = build_chat_request_with_vision(&req, false, true).unwrap();
        // tool message (text only) + follow-up user message carrying the image.
        assert_eq!(wire.messages.len(), 2);
        assert_eq!(wire.messages[0].role, "tool");
        assert_eq!(
            content_text(&wire.messages[0]),
            Some("cat.png:"),
            "tool message stays text-only"
        );
        assert_eq!(wire.messages[1].role, "user");
        match &wire.messages[1].content {
            Some(WireContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(&parts[0], WireContentPart::Text { text }
                    if text.contains("call_1")));
                assert!(matches!(
                    &parts[1],
                    WireContentPart::ImageUrl { image_url }
                        if image_url.url == "data:image/png;base64,iVBORw0KGgoAAEC"
                ));
            }
            other => panic!("expected Parts content, got {other:?}"),
        }
    }

    #[test]
    fn vision_off_tool_result_image_appends_placeholder() {
        // Without this: on a text-only model (DeepSeek's default) a tool's
        // image is still sent as an image part in a follow-up user message,
        // and the request is rejected.
        let tool_result = Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new("call_1"),
                vec![
                    ContentBlock::text("cat.png:"),
                    ContentBlock::Image {
                        source: snaca_core::ImageSource::Base64 {
                            media_type: "image/jpeg".into(),
                            data: "/9j/AAEC".into(),
                        },
                    },
                ],
            )],
        );
        let req = MessageRequest::new("deepseek-chat").with_messages(vec![tool_result]);
        let wire = build_chat_request_with_vision(&req, false, false).unwrap();
        assert_eq!(wire.messages.len(), 1, "no deferred user message");
        match &wire.messages[0].content {
            Some(WireContent::Text(text)) => {
                assert!(text.contains("cat.png:"));
                assert!(text.contains("[image omitted: image/jpeg"));
            }
            other => panic!("expected Text content, got {other:?}"),
        }
    }
}

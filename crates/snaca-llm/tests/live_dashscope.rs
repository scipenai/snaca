//! Live tests against an OpenAI-compatible MaaS deployment that serves
//! Chat Completions, Responses API and an Anthropic-compatible surface
//! (e.g. an Alibaba Cloud DashScope/百炼 dedicated instance). Exercises
//! the three snaca-llm clients end-to-end, including streaming, tool
//! round trips and vision.
//!
//! Base URL conventions differ per client: the chat client appends
//! `/v1/chat/completions` itself (pass the bare host), while the
//! Responses client appends `/responses` (pass the full
//! `.../compatible-mode/v1` root).
//!
//! Skipped by default (`#[ignore]`). Run with:
//!
//! ```bash
//! DASHSCOPE_API_KEY=sk-... \
//! DASHSCOPE_BASE_URL=https://<host>/compatible-mode/v1 \
//! DASHSCOPE_ANTHROPIC_BASE_URL=https://<host>/apps/anthropic \
//! cargo test -p snaca-llm --test live_dashscope -- --ignored --nocapture
//! ```

use futures::StreamExt;
use snaca_core::{ContentBlock, ImageSource, Message};
use snaca_llm::anthropic::AnthropicConfig;
use snaca_llm::deepseek::DeepSeekConfig;
use snaca_llm::openai_responses::OpenAiResponsesConfig;
use snaca_llm::{
    AnthropicClient, ContentDelta, DeepSeekClient, LlmClient, MessageRequest,
    OpenAiResponsesClient, StopReason, StreamEvent, ToolSchema,
};

/// 64x64 solid red PNG — small, deterministic, and above the >10px
/// minimum several vision models enforce on image dimensions.
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";

fn envs() -> (String, String, String, String) {
    let key = std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY env var not set");
    let base = std::env::var("DASHSCOPE_BASE_URL")
        .expect("DASHSCOPE_BASE_URL env var not set (OpenAI-compatible root, e.g. https://<host>/compatible-mode/v1)");
    let anthropic_base = std::env::var("DASHSCOPE_ANTHROPIC_BASE_URL").expect(
        "DASHSCOPE_ANTHROPIC_BASE_URL env var not set (e.g. https://<host>/apps/anthropic)",
    );
    let model = std::env::var("DASHSCOPE_MODEL").unwrap_or_else(|_| "qwen3.8-flash".into());
    (key, base, anthropic_base, model)
}

/// `DeepSeekClient` appends `/v1/chat/completions` itself (mirroring
/// DeepSeek's official SDK, whose base is the bare host), while
/// DashScope's documented OpenAI-compatible base already ends in
/// `/v1`. Strip a trailing `/v1` so both conventions work.
fn chat_base(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

fn render_text(blocks: &[ContentBlock]) -> String {
    let mut s = String::new();
    for b in blocks {
        if let ContentBlock::Text { text } = b {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(text);
        }
    }
    s
}

async fn collect(
    mut stream: futures::stream::BoxStream<'static, snaca_llm::LlmResult<StreamEvent>>,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("stream event"));
    }
    events
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_chat_completions_responds() {
    // Without this: nothing proves DashScope's chat endpoint takes our base
    // URL (bare host, `/v1` stripped), a system message and a plain reply
    // mapped to EndTurn; with a text-only model it is the only chat check.
    let (key, base, _, model) = envs();
    let client = DeepSeekClient::new(
        DeepSeekConfig::new(key)
            .with_model(&model)
            .with_base_url(chat_base(&base)),
    )
    .expect("build client");
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_system("Answer in one short sentence.")
                .with_messages(vec![Message::user_text("What is 2+2?")]),
        )
        .await
        .expect("chat call");
    let text = render_text(&resp.message.content);
    println!("chat text: {text}");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(is_four(&text), "expected the answer 4, got: {text}");
}

/// Models may spell the answer out ("Four") — accept both forms.
fn is_four(text: &str) -> bool {
    text.contains('4') || text.to_lowercase().contains("four")
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_chat_streaming_surfaces_reasoning_then_text() {
    // Without this: DashScope's real `reasoning_content` stream could stop
    // surfacing as Thinking deltas (or end without MessageStop) unnoticed —
    // the unit fixtures only imitate its shape.
    let (key, base, _, model) = envs();
    let client = DeepSeekClient::new(
        DeepSeekConfig::new(key)
            .with_model(&model)
            .with_base_url(chat_base(&base)),
    )
    .expect("build client");
    let stream =
        client
            .create_message_stream(MessageRequest::new(&model).with_messages(vec![
                Message::user_text("9.9和9.11哪个大？先想一想再回答。"),
            ]))
            .await
            .expect("chat stream");
    let events = collect(stream).await;

    let thinking: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Thinking { text },
                ..
            } => Some(text.clone()),
            _ => None,
        })
        .collect();
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
    println!("thinking chars: {}", thinking.chars().count());
    println!("text: {text}");
    assert!(
        !thinking.is_empty(),
        "expected reasoning_content deltas (thinking model)"
    );
    assert!(!text.is_empty(), "expected text deltas");
    assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_anthropic_messages_responds() {
    // Without this: nothing proves DashScope's Anthropic-compatible surface
    // accepts our Messages request and returns an end_turn reply we parse.
    let (key, _, anthropic_base, model) = envs();
    let client = AnthropicClient::new(
        AnthropicConfig::new(key)
            .with_model(&model)
            .with_base_url(anthropic_base),
    )
    .expect("build client");
    // Thinking models burn output tokens on reasoning — budget generously.
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![Message::user_text("What is 2+2? One word answer.")])
                .with_max_tokens(1024),
        )
        .await
        .expect("anthropic call");
    let text = render_text(&resp.message.content);
    println!("anthropic text: {text}");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(is_four(&text), "expected the answer 4, got: {text}");
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_anthropic_sees_image() {
    // Without this: nothing proves DashScope's Anthropic surface accepts our
    // base64 image blocks, or that the model actually sees the pixels.
    let (key, _, anthropic_base, model) = envs();
    let client = AnthropicClient::new(
        AnthropicConfig::new(key)
            .with_model(&model)
            .with_base_url(anthropic_base),
    )
    .expect("build client");
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![red_image_question()])
                .with_max_tokens(1024),
        )
        .await
        .expect("anthropic vision call");
    let text = render_text(&resp.message.content);
    println!("anthropic vision text: {text}");
    assert!(
        recognised_red(&text),
        "model did not recognise the red image: {text}"
    );
}

/// The canonical user message every vision test sends: a caption plus the
/// red PNG. Shared so the three wire paths are compared on identical input.
fn red_image_question() -> Message {
    Message::new(
        snaca_core::Role::User,
        vec![
            ContentBlock::text("这张图片是什么颜色？只回答颜色名。"),
            ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: RED_PNG_BASE64.into(),
                },
            },
        ],
    )
}

fn recognised_red(text: &str) -> bool {
    text.contains('红') || text.to_lowercase().contains("red")
}

fn read_tool_schema() -> ToolSchema {
    ToolSchema {
        name: "Read".into(),
        description: "Read a file from the workspace. Use this whenever the user asks \
                      about a file's contents."
            .into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    }
}

/// History for "the Read tool returned an image": the user's ask, the
/// model's `Read` call, and a `tool_result` whose only block is the red
/// PNG — exactly what `ReadTool` produces for an image file.
fn image_tool_result_history(call_id: &str) -> Vec<Message> {
    vec![
        Message::user_text("用 Read 工具读一下 cat.png，然后告诉我它是什么颜色？只回答颜色名。"),
        Message::new(
            snaca_core::Role::Assistant,
            vec![ContentBlock::tool_use(
                call_id,
                "Read",
                serde_json::json!({"path": "cat.png"}),
            )],
        ),
        Message::new(
            snaca_core::Role::Tool,
            vec![ContentBlock::tool_result(
                snaca_core::ToolUseId::new(call_id),
                vec![ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: RED_PNG_BASE64.into(),
                    },
                }],
            )],
        ),
    ]
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_chat_sees_image() {
    // Without this: nothing proves DashScope's chat endpoint accepts our
    // `image_url` data-URL parts, or that the model actually sees them. That
    // path (`WireContent::Parts`) only engages with `with_vision(true)`: the
    // DeepSeek client defaults to text-only, as DeepSeek's own models are.
    let (key, base, _, model) = envs();
    let client = DeepSeekClient::new(
        DeepSeekConfig::new(key)
            .with_model(&model)
            .with_base_url(chat_base(&base))
            .with_vision(true),
    )
    .expect("build client");
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![red_image_question()])
                .with_max_tokens(1024),
        )
        .await
        .expect("chat vision call");
    let text = render_text(&resp.message.content);
    println!("chat vision text: {text}");
    assert!(
        recognised_red(&text),
        "model did not recognise the red image: {text}"
    );
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_responses_sees_image() {
    // Without this: nothing proves DashScope's Responses endpoint accepts our
    // `input_image` data URLs (vision is on by default), or that the model
    // actually sees the pixels.
    let (key, base, _, model) = envs();
    let client = OpenAiResponsesClient::new(
        OpenAiResponsesConfig::new(key)
            .with_model(&model)
            .with_base_url(base),
    )
    .expect("build client");
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![red_image_question()])
                .with_max_tokens(1024),
        )
        .await
        .expect("responses vision call");
    let text = render_text(&resp.message.content);
    println!("responses vision text: {text}");
    assert!(
        recognised_red(&text),
        "model did not recognise the red image: {text}"
    );
}

/// Concatenated text deltas from a collected stream.
fn stream_text(events: &[StreamEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Text { text },
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_chat_streams_tool_returned_image() {
    // Without this: nothing proves DashScope's streaming chat endpoint (what
    // the engine uses) accepts our own construction for tool-returned
    // pixels — OpenAI tool messages are text-only, so they go in a follow-up
    // user message: assistant(tool_calls) → tool → user(image_url) — or that
    // the model actually sees them rather than just tolerating the shape.
    let (key, base, _, model) = envs();
    let client = DeepSeekClient::new(
        DeepSeekConfig::new(key)
            .with_model(&model)
            .with_base_url(chat_base(&base))
            .with_vision(true),
    )
    .expect("build client");
    let stream = client
        .create_message_stream(
            MessageRequest::new(&model)
                .with_messages(image_tool_result_history("call_img_1"))
                .with_tools(vec![read_tool_schema()])
                .with_max_tokens(1024),
        )
        .await
        .expect("chat streaming vision call");
    let text = stream_text(&collect(stream).await);
    println!("chat streaming tool-image text: {text}");
    assert!(
        recognised_red(&text),
        "model did not see the tool-returned image while streaming: {text}"
    );
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_responses_streams_tool_returned_image() {
    // Without this: nothing proves DashScope's streaming Responses endpoint
    // accepts the same deferred-image workaround — function_call →
    // function_call_output (text-only) → user(input_image) — or that the
    // model sees the pixels a Read call returned.
    let (key, base, _, model) = envs();
    let client = OpenAiResponsesClient::new(
        OpenAiResponsesConfig::new(key)
            .with_model(&model)
            .with_base_url(base),
    )
    .expect("build client");
    let stream = client
        .create_message_stream(
            MessageRequest::new(&model)
                .with_messages(image_tool_result_history("call_img_1"))
                .with_tools(vec![read_tool_schema()])
                .with_max_tokens(1024),
        )
        .await
        .expect("responses streaming vision call");
    let text = stream_text(&collect(stream).await);
    println!("responses streaming tool-image text: {text}");
    assert!(
        recognised_red(&text),
        "model did not see the tool-returned image while streaming: {text}"
    );
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_anthropic_streams_tool_returned_image() {
    // Without this: nothing proves DashScope's streaming Anthropic surface
    // accepts image blocks inside tool_result content (no follow-up user
    // message needed there), or that the model sees them — the converter once
    // filtered them out and the model got an empty tool_result.
    let (key, _, anthropic_base, model) = envs();
    let client = AnthropicClient::new(
        AnthropicConfig::new(key)
            .with_model(&model)
            .with_base_url(anthropic_base),
    )
    .expect("build client");
    let stream = client
        .create_message_stream(
            MessageRequest::new(&model)
                .with_messages(image_tool_result_history("call_img_1"))
                .with_tools(vec![read_tool_schema()])
                .with_max_tokens(1024),
        )
        .await
        .expect("anthropic streaming vision call");
    let text = stream_text(&collect(stream).await);
    println!("anthropic streaming tool-image text: {text}");
    assert!(
        recognised_red(&text),
        "model did not see the tool-returned image while streaming: {text}"
    );
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_responses_responds() {
    // Without this: nothing proves DashScope accepts a tool-less Responses
    // request (what compaction and memory extraction send) and that its reply
    // maps to EndTurn; every other Responses check here offers tools or needs
    // vision.
    let (key, base, _, model) = envs();
    let client = OpenAiResponsesClient::new(
        OpenAiResponsesConfig::new(key)
            .with_model(&model)
            .with_base_url(base),
    )
    .expect("build client");
    let resp = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![Message::user_text("What is 2+2? One word answer.")]),
        )
        .await
        .expect("responses call");
    let text = render_text(&resp.message.content);
    println!("responses text: {text}");
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert!(is_four(&text), "expected the answer 4, got: {text}");
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_responses_streaming_surfaces_reasoning_and_text() {
    // Without this: DashScope's real reasoning event names could stop mapping
    // to Thinking deltas, or usage on `response.completed` could be lost,
    // unnoticed — OpenAI streams no reasoning text to check against.
    let (key, base, _, model) = envs();
    let client = OpenAiResponsesClient::new(
        OpenAiResponsesConfig::new(key)
            .with_model(&model)
            .with_base_url(base),
    )
    .expect("build client");
    let stream =
        client
            .create_message_stream(MessageRequest::new(&model).with_messages(vec![
                Message::user_text("9.9和9.11哪个大？先想一想再回答。"),
            ]))
            .await
            .expect("responses stream");
    let events = collect(stream).await;

    let has_thinking = events.iter().any(|e| {
        matches!(
            e,
            StreamEvent::ContentBlockDelta {
                delta: ContentDelta::Thinking { .. },
                ..
            }
        )
    });
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
    let usage = events.iter().find_map(|e| match e {
        StreamEvent::MessageDelta { usage, .. } => usage.clone(),
        _ => None,
    });
    println!("text: {text}");
    println!("usage: {usage:?}");
    assert!(has_thinking, "expected reasoning deltas on the stream");
    assert!(!text.is_empty(), "expected text deltas");
    assert!(usage.is_some(), "expected usage on response.completed");
    assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn dashscope_responses_tool_call_round_trip() {
    // Without this: nothing proves DashScope accepts our flat `strict: false`
    // tools, returns a function_call we parse, and accepts the replayed
    // function_call + function_call_output on the next request.
    let (key, base, _, model) = envs();
    let client = OpenAiResponsesClient::new(
        OpenAiResponsesConfig::new(key)
            .with_model(&model)
            .with_base_url(base),
    )
    .expect("build client");
    let read_tool = read_tool_schema();
    let first = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![Message::user_text("用 Read 工具读一下 notes.txt")])
                .with_tools(vec![read_tool.clone()]),
        )
        .await
        .expect("responses tool call");
    let tool_use = first.message.content.iter().find_map(|b| match b {
        ContentBlock::ToolUse { id, name, input } => {
            Some((id.clone(), name.clone(), input.clone()))
        }
        _ => None,
    });
    let (call_id, name, input) = tool_use.expect("expected a function_call for the Read tool");
    println!("tool call: {name} {input}");
    assert_eq!(name, "Read");

    // Feed the tool result back and confirm the loop completes.
    let second = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![
                    Message::user_text("用 Read 工具读一下 notes.txt"),
                    Message::new(
                        snaca_core::Role::Assistant,
                        vec![ContentBlock::tool_use(call_id.as_str(), &name, input)],
                    ),
                    Message::new(
                        snaca_core::Role::Tool,
                        vec![ContentBlock::tool_result(
                            call_id,
                            vec![ContentBlock::text("SNACA is an IM-first agent server.")],
                        )],
                    ),
                ])
                .with_tools(vec![read_tool]),
        )
        .await
        .expect("responses tool follow-up");
    let text = render_text(&second.message.content);
    println!("follow-up text: {text}");
    assert_eq!(second.stop_reason, StopReason::EndTurn);
    assert!(
        text.to_lowercase().contains("snaca"),
        "model did not use the tool result: {text}"
    );
}

//! Offline end-to-end vision: real workspace → real `Read` tool → real
//! provider client → HTTP, with a mock server standing in for the model.
//!
//! `live_vision.rs` proves a real model sees the pixels but needs network
//! credentials, so it never runs in CI. This suite runs the same chain
//! offline and asserts on what actually reaches the wire:
//!
//! ```text
//! Engine::handle_turn
//!   → Read on a 4.8 MB BMP          (image_prep: BMP → downscaled JPEG)
//!   → persisted to SQLite
//!   → load_history / byte cap / pairing
//!   → provider converter            (send-time gate)
//!   → HTTP request body             ← asserted here
//! ```
//!
//! What this guards is the *chain*: an image Read captures must survive
//! persistence, the history window and the provider converter, and reach
//! the request body as the normalised JPEG. The fixture would be
//! re-encoded for its size or its format alone, so it can't tell those
//! rules apart — each is pinned by `snaca-tools`' `image_prep` unit tests
//! (e.g. a *small* BMP still converted), not here.
//!
//! Two legs: Chat Completions at the default budget, Anthropic under a
//! configured edge cap. The Responses wire adds nothing to the chain; its
//! tool-result images are pinned by `snaca-llm`'s `openai_responses`
//! converter and HTTP tests.
//!
//! The engine always streams; the mock speaks plain JSON, so the real
//! client is wrapped in [`NonStreaming`], which drops back to the
//! trait's synthesized stream. Everything up to and including the HTTP
//! request is the production code path.

use async_trait::async_trait;
use data_encoding::BASE64;
use image::{ImageFormat, Rgb, RgbImage};
use serde_json::{json, Value};
use snaca_core::{ContentBlock, ImageSource, ProjectId, TenantId, ThreadId};
use snaca_engine::{Engine, EngineConfig, TurnRequest};
use snaca_llm::{
    AnthropicClient, AnthropicConfig, DeepSeekClient, DeepSeekConfig, LlmClient, LlmResult,
    MessageRequest, MessageResponse, ProviderCaps,
};
use snaca_state::Database;
use snaca_tools::default_m1_registry;
use snaca_workspace::WorkspaceLayout;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Wraps a real provider client but routes streaming through
/// `create_message`, so the mock can answer with a plain JSON body.
struct NonStreaming(Arc<dyn LlmClient>);

#[async_trait]
impl LlmClient for NonStreaming {
    fn provider_name(&self) -> &'static str {
        self.0.provider_name()
    }

    fn model(&self) -> &str {
        self.0.model()
    }

    fn capabilities(&self) -> ProviderCaps {
        ProviderCaps {
            streaming: false,
            ..self.0.capabilities()
        }
    }

    async fn create_message(&self, request: MessageRequest) -> LlmResult<MessageResponse> {
        self.0.create_message(request).await
    }
}

/// First request → a `Read` tool call on `photo.bmp`; every later
/// request → a final text answer.
struct Script {
    calls: AtomicUsize,
    tool_call: Value,
    final_text: Value,
}

impl Respond for Script {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let body = match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => &self.tool_call,
            _ => &self.final_text,
        };
        ResponseTemplate::new(200).set_body_json(body)
    }
}

fn anthropic_script() -> Script {
    let msg = |content: Value, stop: &str| {
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "mock",
            "content": content, "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 5},
        })
    };
    Script {
        calls: AtomicUsize::new(0),
        tool_call: msg(
            json!([{"type": "tool_use", "id": "toolu_1", "name": "Read",
                    "input": {"path": "photo.bmp"}}]),
            "tool_use",
        ),
        final_text: msg(json!([{"type": "text", "text": "红色"}]), "end_turn"),
    }
}

fn chat_script() -> Script {
    let completion = |message: Value, finish: &str| {
        json!({
            "id": "chat_1", "object": "chat.completion", "model": "mock",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
    };
    Script {
        calls: AtomicUsize::new(0),
        tool_call: completion(
            json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "Read", "arguments": "{\"path\":\"photo.bmp\"}"}}]}),
            "tool_calls",
        ),
        final_text: completion(json!({"role": "assistant", "content": "红色"}), "stop"),
    }
}

/// 2000x800 solid-red 24-bit BMP (~4.8 MB) — over both the byte budget and
/// the edge cap.
fn red_bmp() -> Vec<u8> {
    let img = RgbImage::from_pixel(2000, 800, Rgb([220, 20, 20]));
    let mut out = Cursor::new(Vec::new());
    img.write_to(&mut out, ImageFormat::Bmp).unwrap();
    out.into_inner()
}

/// Every inline image payload in a request body, as `(media_type, base64)`,
/// whatever the wire shape: Anthropic `{media_type, data}` sources or
/// OpenAI-family `data:` URLs.
fn wire_images(body: &Value, out: &mut Vec<(String, String)>) {
    match body {
        Value::Object(map) => {
            if let (Some(Value::String(mt)), Some(Value::String(data))) =
                (map.get("media_type"), map.get("data"))
            {
                out.push((mt.clone(), data.clone()));
            }
            map.values().for_each(|v| wire_images(v, out));
        }
        Value::Array(items) => items.iter().for_each(|v| wire_images(v, out)),
        Value::String(s) => {
            if let Some((mt, data)) = s
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(";base64,"))
            {
                out.push((mt.to_string(), data.to_string()));
            }
        }
        _ => {}
    }
}

async fn run(client: Arc<dyn LlmClient>, server: &MockServer) {
    run_with(
        client,
        server,
        EngineConfig::default_for("mock"),
        (1568, 627),
    )
    .await;
}

/// Drive one Read-an-image turn under `config` and assert the image on
/// the wire has exactly `expected_dims`.
async fn run_with(
    client: Arc<dyn LlmClient>,
    server: &MockServer,
    config: EngineConfig,
    expected_dims: (u32, u32),
) {
    let tmp = tempfile::tempdir().unwrap();
    let layout = WorkspaceLayout::new(tmp.path()).unwrap();
    let tenant = TenantId::new("tenant_vision");
    let project = ProjectId::from_raw("proj_vision");
    layout.ensure_project(&tenant, &project).unwrap();
    std::fs::write(
        layout.workspace_dir(&tenant, &project).join("photo.bmp"),
        red_bmp(),
    )
    .unwrap();

    let db = Database::open_in_memory().await.unwrap();
    let engine = Engine::new(
        Arc::new(NonStreaming(client)),
        default_m1_registry(),
        db.clone(),
        layout,
        config,
    );
    let thread = ThreadId::new("chat_vision");
    let outcome = engine
        .handle_turn(TurnRequest {
            tenant_id: tenant,
            project_id: project,
            thread_id: thread.clone(),
            user_text: "读一下 photo.bmp，这是什么颜色？".into(),
            message_id: None,
            ephemeral_system: None,
        })
        .await
        .expect("turn");
    assert_eq!(outcome.assistant_text, "红色");
    assert_eq!(outcome.iterations, 2, "one tool round trip");

    // Persisted: the re-encoded JPEG plus the note — never the raw BMP.
    let msgs = db.recent_messages(&thread, 20).await.unwrap();
    let result: Vec<&ContentBlock> = msgs
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolResult { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(
        matches!(result.as_slice(), [
            ContentBlock::Text { text },
            ContentBlock::Image { source: ImageSource::Base64 { media_type, .. } },
        ] if text.contains("bmp 2000x800") && media_type == "image/jpeg"),
        "persisted tool result: {result:?}"
    );

    // On the wire: the follow-up request carries exactly that JPEG,
    // within the edge cap, and still red.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let body: Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert!(
        body.to_string().contains("bmp 2000x800"),
        "re-encode note reached the model"
    );
    let mut images = Vec::new();
    wire_images(&body, &mut images);
    assert_eq!(images.len(), 1, "exactly one image on the wire");
    let (media_type, data) = &images[0];
    assert_eq!(media_type, "image/jpeg");
    let bytes = BASE64.decode(data.as_bytes()).unwrap();
    assert_eq!(
        image::guess_format(&bytes).unwrap(),
        ImageFormat::Jpeg,
        "payload is really JPEG"
    );
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert_eq!(
        decoded.dimensions(),
        expected_dims,
        "downscaled, aspect kept"
    );
    let [r, g, b] = decoded
        .get_pixel(expected_dims.0 / 2, expected_dims.1 / 2)
        .0;
    assert!(r > 180 && g < 60 && b < 60, "still red: {:?}", (r, g, b));
}

#[tokio::test]
async fn chat_completions_wire_carries_reencoded_image() {
    // Without this: a vision-enabled Chat Completions endpoint got a Read
    // image as "[image omitted]" text — the client's vision flag or the
    // deferred user-message image never reached the request body.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(chat_script())
        .mount(&server)
        .await;
    let client = DeepSeekClient::new(
        DeepSeekConfig::new("test-key")
            .with_base_url(server.uri())
            .with_model("mock")
            .with_vision(true),
    )
    .unwrap();
    run(Arc::new(client), &server).await;
}

#[tokio::test]
async fn configured_image_edge_reaches_the_read_tool() {
    // Without this: [engine] image_max_edge_px was ignored and Read always
    // downscaled to the 1568 px default, whatever the operator set. Also
    // the suite's Anthropic leg: the image must reach a Messages request.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(anthropic_script())
        .mount(&server)
        .await;
    let client = AnthropicClient::new(
        AnthropicConfig::new("test-key")
            .with_base_url(server.uri())
            .with_model("mock"),
    )
    .unwrap();
    let config = EngineConfig {
        image_max_edge_px: 800,
        ..EngineConfig::default_for("mock")
    };
    run_with(Arc::new(client), &server, config, (800, 320)).await;
}

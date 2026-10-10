//! End-to-end vision: real workspace → real `Read` tool → real provider.
//!
//! The `snaca-llm` unit tests prove the converters emit the right JSON;
//! `live_dashscope.rs` proves a provider accepts that JSON and the model
//! sees the pixels; `vision_pipeline.rs` runs the production chain offline
//! against a mock. This is the one that runs that chain against a real
//! model:
//!
//! ```text
//! Engine::handle_turn
//!   → Read tool on a real image file  (ContentBlock::Image)
//!   → persisted to SQLite
//!   → load_history / ensure_tool_result_pairing
//!   → provider converter
//!   → model
//! ```
//!
//! That chain is where the byte-accounting work and the vision work meet:
//! an image block has to survive persistence, the history window, the
//! pairing gate and the per-result clamp without being filtered, cleared
//! or truncated. This test asserts both ends — the `Image` block is in
//! the DB, and the model names the colour it depicts.
//!
//! Skipped by default (`#[ignore]`). Run with:
//!
//! ```bash
//! DASHSCOPE_API_KEY=sk-... \
//! DASHSCOPE_BASE_URL=https://<host>/compatible-mode/v1 \
//! cargo test -p snaca-engine --test live_vision -- --ignored --nocapture
//! ```

use snaca_core::{ContentBlock, ImageSource, ProjectId, TenantId, ThreadId};
use snaca_engine::{Engine, EngineConfig, TurnRequest};
use snaca_llm::{DeepSeekClient, DeepSeekConfig};
use snaca_state::Database;
use snaca_tools::default_m1_registry;
use snaca_workspace::WorkspaceLayout;
use std::sync::Arc;

fn envs() -> (String, String, String) {
    let key = std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY env var not set");
    let base = std::env::var("DASHSCOPE_BASE_URL")
        .expect("DASHSCOPE_BASE_URL env var not set (OpenAI-compatible root)");
    let model = std::env::var("DASHSCOPE_MODEL").unwrap_or_else(|_| "qwen3.8-flash".into());
    (key, base, model)
}

/// `DeepSeekClient` appends `/v1/chat/completions` itself; DashScope's
/// documented base already ends in `/v1`. Strip it so both work.
fn chat_base(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    trimmed.strip_suffix("/v1").unwrap_or(trimmed).to_string()
}

/// One live turn: drop `bytes` into the workspace as `file`, ask the
/// model (via Read) what colour it is, and assert it answers red. Returns
/// the media type of the image block persisted in the tool result.
async fn read_and_ask(file: &str, bytes: Vec<u8>) -> String {
    let (key, base, model) = envs();

    let tmp = tempfile::tempdir().unwrap();
    let layout = WorkspaceLayout::new(tmp.path()).unwrap();
    let tenant = TenantId::new("tenant_vision");
    let project = ProjectId::from_raw("proj_vision");
    layout.ensure_project(&tenant, &project).unwrap();
    std::fs::write(layout.workspace_dir(&tenant, &project).join(file), bytes).unwrap();

    let db = Database::open_in_memory().await.unwrap();
    // Vision must be switched on explicitly: the DeepSeek client defaults
    // to text-only, which is what a stock config would use against
    // DeepSeek's own (text-only) hosted models.
    let llm = Arc::new(
        DeepSeekClient::new(
            DeepSeekConfig::new(key)
                .with_model(&model)
                .with_base_url(chat_base(&base))
                .with_vision(true),
        )
        .expect("build client"),
    );

    let engine = Engine::new(
        llm,
        default_m1_registry(),
        db.clone(),
        layout,
        EngineConfig::default_for(&model),
    );

    let thread = ThreadId::new("chat_vision");
    let outcome = engine
        .handle_turn(TurnRequest {
            tenant_id: tenant.clone(),
            project_id: project.clone(),
            thread_id: thread.clone(),
            user_text: format!(
                "用 Read 工具读一下 {file}，然后告诉我这张图是什么颜色？只回答颜色名。"
            ),
            message_id: None,
            ephemeral_system: None,
        })
        .await
        .expect("turn");

    println!("iterations: {}", outcome.iterations);
    println!("assistant: {}", outcome.assistant_text);

    // End 1 — the engine actually persisted pixels, not a placeholder or
    // an empty tool_result (the shape this work fixed).
    let msgs = db.recent_messages(&thread, 20).await.unwrap();
    let mut media_types: Vec<String> = msgs
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolResult { content, .. } => Some(content),
            _ => None,
        })
        .flat_map(|c| c.iter())
        .filter_map(|b| match b {
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, .. },
            } => Some(media_type.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        media_types.len(),
        1,
        "expected exactly one Image block in the persisted tool result"
    );

    // End 2 — the model received those pixels through the whole chain.
    assert!(
        outcome.assistant_text.contains('红')
            || outcome.assistant_text.to_lowercase().contains("red"),
        "model did not describe the image as red: {}",
        outcome.assistant_text
    );
    assert!(
        outcome.iterations >= 2,
        "expected a tool round trip, got {} iteration(s)",
        outcome.iterations
    );
    media_types.remove(0)
}

#[tokio::test]
#[ignore = "requires DASHSCOPE_API_KEY; live network call"]
async fn engine_reencodes_a_bmp_and_the_model_still_sees_it() {
    // Without this: the JPEG Read re-encodes a BMP into could be refused
    // by, or illegible to, a real model — vision_pipeline.rs only decodes
    // it locally. A 2000x800 BMP is refused by every provider as-is and is
    // over the edge cap, so Read must downscale and re-encode it.
    let img = image::RgbImage::from_pixel(2000, 800, image::Rgb([220, 20, 20]));
    let mut bmp = std::io::Cursor::new(Vec::new());
    img.write_to(&mut bmp, image::ImageFormat::Bmp).unwrap();
    let media_type = read_and_ask("red.bmp", bmp.into_inner()).await;
    assert_eq!(media_type, "image/jpeg");
}

//! Unit-style tests for `collapse_old_tool_results` — the helper that
//! shrinks oversized read-only tool_results in older history slots.
//!
//! The full pipeline (DB → load_history → LLM) is exercised by the
//! existing `compaction.rs` and `turn_loop.rs` integration tests; here
//! we just nail down the rewrite semantics:
//!
//! - kept-tail messages stay verbatim
//! - small tool_results stay verbatim regardless of age
//! - large tool_results from collapsible tools shrink to a marker
//! - large tool_results from *unknown* tools stay verbatim (safe default)
//! - errors never collapse
//! - only text collapses — images beside it stay
//! - threshold = 0 disables collapse entirely

use chrono::Utc;
use serde_json::json;
use snaca_core::{ContentBlock, Message, MessageId, Role, ToolUseId};
use snaca_engine::engine::collapse_old_tool_results;

fn user(text: &str) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::User,
        content: vec![ContentBlock::text(text)],
        created_at: Utc::now(),
    }
}

fn assistant_calling(tool_use_id: &str, name: &str) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::Assistant,
        content: vec![
            ContentBlock::text("calling..."),
            ContentBlock::tool_use(tool_use_id, name, json!({"path": "foo.rs"})),
        ],
        created_at: Utc::now(),
    }
}

fn tool_result(tool_use_id: &str, body: &str) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::Tool,
        content: vec![ContentBlock::tool_result(
            ToolUseId::new(tool_use_id),
            vec![ContentBlock::text(body)],
        )],
        created_at: Utc::now(),
    }
}

fn tool_error(tool_use_id: &str, message: &str) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::Tool,
        content: vec![ContentBlock::tool_error(
            ToolUseId::new(tool_use_id),
            message,
        )],
        created_at: Utc::now(),
    }
}

fn assistant_text(text: &str) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::Assistant,
        content: vec![ContentBlock::text(text)],
        created_at: Utc::now(),
    }
}

/// Helper: get the inner text of a tool_result message (panics if the
/// shape doesn't match — tests should always use the constructors above).
fn tool_result_text(m: &Message) -> &str {
    match &m.content[0] {
        ContentBlock::ToolResult { content, .. } => match &content[0] {
            ContentBlock::Text { text } => text,
            other => panic!("expected text content, got {other:?}"),
        },
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[test]
fn large_read_result_in_old_position_gets_collapsed() {
    let big = "x".repeat(2048);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        tool_result("call-1", &big),
        assistant_text("done"),
        // Kept-tail pair:
        user("now do something else"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    // Tool result at index 2 is *before* the kept-tail (last 2),
    // collapsible tool, body well over 1024 → should be a marker.
    let collapsed = tool_result_text(&out[2]);
    assert!(
        collapsed.starts_with("<Read result:"),
        "expected collapse marker, got: {collapsed}"
    );
    assert!(collapsed.contains("2048 bytes"), "got: {collapsed}");
}

#[test]
fn small_result_stays_verbatim() {
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        tool_result("call-1", "small body under 1KB"),
        assistant_text("ack"),
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    let body = tool_result_text(&out[2]);
    assert_eq!(body, "small body under 1KB");
}

#[test]
fn kept_tail_never_collapses_even_when_large() {
    let big = "x".repeat(4096);
    // The tool_result is in the kept tail (last 2 messages).
    let messages = vec![
        user("hi"),
        assistant_text("ack"),
        assistant_calling("call-1", "Read"),
        tool_result("call-1", &big),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    let body = tool_result_text(&out[3]);
    assert_eq!(
        body.len(),
        4096,
        "tail tool_result must stay verbatim; got {} bytes",
        body.len()
    );
}

#[test]
fn unknown_tool_results_never_collapse() {
    let big = "y".repeat(4096);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "MysteryMcpTool"),
        tool_result("call-1", &big),
        assistant_text("done"),
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    // MysteryMcpTool isn't in COLLAPSIBLE_TOOL_NAMES — safer to keep
    // verbatim than risk hiding a write side effect.
    let body = tool_result_text(&out[2]);
    assert_eq!(body.len(), 4096);
}

#[test]
fn errors_never_collapse() {
    let big_err = "stack trace ".repeat(200);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        tool_error("call-1", &big_err),
        assistant_text("oops"),
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    // Error message stays as-is — even oversized errors are usually
    // signal-heavy.
    let body = tool_result_text(&out[2]);
    assert_eq!(body.len(), big_err.len());
}

#[test]
fn threshold_zero_disables_collapse() {
    let big = "x".repeat(8192);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        tool_result("call-1", &big),
        assistant_text("done"),
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 0);

    let body = tool_result_text(&out[2]);
    assert_eq!(body.len(), 8192, "threshold=0 should be a no-op");
}

/// keep_recent that covers the entire vec → nothing to collapse.
#[test]
fn keep_recent_larger_than_history_is_safe_noop() {
    let big = "x".repeat(4096);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        tool_result("call-1", &big),
    ];

    let out = collapse_old_tool_results(messages, 10, 1024);

    let body = tool_result_text(&out[2]);
    assert_eq!(body.len(), 4096);
}

/// Compaction summariser passes `keep_recent = 0` (the kept tail has
/// already been sliced off upstream). Everything older than the cutoff
/// is fair game.
#[test]
fn keep_recent_zero_collapses_everything_eligible() {
    let big = "x".repeat(4096);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Grep"),
        tool_result("call-1", &big),
        assistant_text("done"),
    ];

    let out = collapse_old_tool_results(messages, 0, 1024);

    let body = tool_result_text(&out[2]);
    assert!(
        body.starts_with("<Grep result:"),
        "expected Grep marker, got: {body}"
    );
}

fn read_result_with_image(tool_use_id: &str, text: &str, b64_len: usize) -> Message {
    Message {
        id: MessageId::new(),
        role: Role::Tool,
        content: vec![ContentBlock::tool_result(
            ToolUseId::new(tool_use_id),
            vec![ContentBlock::text(text), png(b64_len)],
        )],
        created_at: Utc::now(),
    }
}

fn png(b64_len: usize) -> ContentBlock {
    ContentBlock::Image {
        source: snaca_core::ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "A".repeat(b64_len),
        },
    }
}

fn tool_result_body(m: &Message) -> &[ContentBlock] {
    match &m.content[0] {
        ContentBlock::ToolResult { content, .. } => content,
        other => panic!("expected ToolResult, got {other:?}"),
    }
}

#[test]
fn old_image_result_is_left_to_the_image_defences() {
    // Without this: collapse would weigh the pixels, so an image the
    // current turn is still working with — a long tool loop pushes it out
    // of the `keep_recent` tail — would be elided, and the model would
    // answer about a picture it can no longer see. Retiring images is the
    // job of `retire_stale_tool_images` and the byte cap, which know the
    // turn boundary.
    let messages = vec![
        user("compare photo.png with the next one"),
        assistant_calling("call-1", "Read"),
        read_result_with_image("call-1", "png 64x64", 4096),
        assistant_calling("call-2", "Read"),
        // Kept-tail pair:
        tool_result("call-2", "small"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    let body = tool_result_body(&out[2]);
    assert!(
        matches!(body, [ContentBlock::Text { text }, ContentBlock::Image { .. }] if text == "png 64x64"),
        "image result must stay verbatim, got: {body:?}"
    );
}

#[test]
fn collapsing_large_text_keeps_the_images_beside_it() {
    // Without this: eliding a big text body would take its images with it,
    // bypassing the newest-3 rule and the turn boundary that govern them.
    let big = "x".repeat(2048);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Read"),
        read_result_with_image("call-1", &big, 4096),
        assistant_text("done"),
        // Kept-tail pair:
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024);

    let body = tool_result_body(&out[2]);
    assert!(
        matches!(body, [ContentBlock::Text { text }, ContentBlock::Image { .. }]
            if text.starts_with("<Read result: 2048 bytes elided")),
        "text collapsed, image kept: {body:?}"
    );
}

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

    let out = collapse_old_tool_results(messages, 2, 1024, false);

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

    let out = collapse_old_tool_results(messages, 2, 1024, false);

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

    let out = collapse_old_tool_results(messages, 2, 1024, false);

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

    let out = collapse_old_tool_results(messages, 2, 1024, false);

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

    let out = collapse_old_tool_results(messages, 2, 1024, false);

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

    let out = collapse_old_tool_results(messages, 2, 0, false);

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

    let out = collapse_old_tool_results(messages, 10, 1024, false);

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

    let out = collapse_old_tool_results(messages, 0, 1024, false);

    let body = tool_result_text(&out[2]);
    assert!(
        body.starts_with("<Grep result:"),
        "expected Grep marker, got: {body}"
    );
}

/// `force_all_tools = true` collapses a large Bash result even though
/// Bash isn't in COLLAPSIBLE_TOOL_NAMES — the no-compaction load path
/// relies on this to tame big file-extraction dumps.
#[test]
fn force_all_tools_collapses_bash() {
    let big = "x".repeat(4096);
    let messages = vec![
        user("hi"),
        assistant_calling("call-1", "Bash"),
        tool_result("call-1", &big),
        assistant_text("done"),
        user("next"),
        assistant_text("ok"),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024, true);

    let collapsed = tool_result_text(&out[2]);
    assert!(
        collapsed.starts_with("<Bash result:"),
        "expected Bash marker under force_all_tools, got: {collapsed}"
    );
    assert!(collapsed.contains("4096 bytes"), "got: {collapsed}");
}

/// A Skill result stays verbatim with the conservative whitelist
/// (`false`) but collapses under `force_all_tools = true`.
#[test]
fn force_all_tools_collapses_skill_but_default_keeps_verbatim() {
    let big = "z".repeat(4096);
    let build = || {
        vec![
            user("hi"),
            assistant_calling("call-1", "Skill"),
            tool_result("call-1", &big),
            assistant_text("done"),
            user("next"),
            assistant_text("ok"),
        ]
    };

    let default_out = collapse_old_tool_results(build(), 2, 1024, false);
    assert_eq!(
        tool_result_text(&default_out[2]).len(),
        4096,
        "Skill must stay verbatim under the conservative whitelist"
    );

    let forced_out = collapse_old_tool_results(build(), 2, 1024, true);
    assert!(
        tool_result_text(&forced_out[2]).starts_with("<Skill result:"),
        "Skill must collapse under force_all_tools"
    );
}

/// Under `force_all_tools`, the most recent tool_result (in the kept
/// tail) stays verbatim while an older Bash dump collapses.
#[test]
fn recent_tool_results_stay_verbatim_old_ones_collapse() {
    let old_big = "a".repeat(4096);
    let new_big = "b".repeat(4096);
    let messages = vec![
        user("start"),
        assistant_calling("call-old", "Bash"),
        tool_result("call-old", &old_big),
        assistant_text("mid"),
        // Kept tail (last 2): the recent Bash round-trip.
        assistant_calling("call-new", "Bash"),
        tool_result("call-new", &new_big),
    ];

    let out = collapse_old_tool_results(messages, 2, 1024, true);

    let old_body = tool_result_text(&out[2]);
    assert!(
        old_body.starts_with("<Bash result:"),
        "old Bash result should collapse, got: {old_body}"
    );
    let new_body = tool_result_text(&out[5]);
    assert_eq!(
        new_body.len(),
        4096,
        "recent Bash result must stay verbatim"
    );
}

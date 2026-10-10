//! Auto-compaction integration tests.
//!
//! The engine compacts a thread when a single LLM round trip's *input*
//! tokens crosses [`EngineConfig::compact_after_input_tokens`]. After
//! compaction, the next turn's history is the synthetic summary preamble
//! plus the kept tail (`compact_keep_recent` most recent messages).
//!
//! These tests use a scripted `MockLlmClient`. The summarisation request
//! the engine fires after a turn lands is itself an LLM call that the
//! mock has to answer — so for every turn that crosses the threshold we
//! enqueue *two* responses: the turn's terminal assistant message + the
//! summariser's reply.

use snaca_core::{
    ContentBlock, ImageSource, Message, MessageId, ProjectId, Role, SessionId, TenantId, ThreadId,
    ToolUseId, Usage,
};
use snaca_engine::{Engine, EngineConfig, TurnRequest};
use snaca_llm::{MessageResponse, StopReason};
use snaca_state::{Database, NewMessage, NewThread};
use snaca_tools_api::ToolRegistryBuilder;
use snaca_workspace::WorkspaceLayout;
use std::sync::Arc;

mod common;
use common::{assistant_text, EchoTool, MockLlmClient};

fn assistant_text_with_input_tokens(text: &str, input_tokens: u64) -> MessageResponse {
    MessageResponse {
        id: "mock".into(),
        message: Message {
            id: MessageId::new(),
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
            created_at: chrono::Utc::now(),
        },
        usage: Usage {
            input_tokens,
            output_tokens: 1,
            ..Default::default()
        },
        stop_reason: StopReason::EndTurn,
    }
}

struct Fixture {
    engine: Engine,
    db: Database,
    llm: Arc<MockLlmClient>,
    _tmp: tempfile::TempDir,
}

/// Build an engine wired with `compact_after_input_tokens=100` and
/// `compact_keep_recent=2` — extreme values so a couple of toy turns
/// trigger compaction.
///
/// `compact_blocking = true` so `handle_turn` awaits compaction before
/// returning. The tests assert on `get_thread_summary` immediately
/// after the turn lands and the scripted `MockLlmClient` queues the
/// summariser reply in lock-step with the turn's terminal response;
/// background-mode compaction would race both invariants. The default
/// production path is fire-and-forget — covered by `compaction_runs_in_background`.
async fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = Some(100);
    cfg.compact_keep_recent = 2;
    // Legacy "compress from the beginning" — these tests pre-date
    // first-N protection and assert on a "summary at the head" prompt
    // shape. New behaviour is exercised separately in
    // `protect_first_n_preserves_head_messages`.
    cfg.protect_first_n = 0;
    cfg.compact_blocking = true;
    let engine = Engine::new(llm.clone(), tools, db.clone(), workspace, cfg);
    Fixture {
        engine,
        db,
        llm,
        _tmp: tmp,
    }
}

fn turn_request(thread: &str, user_text: &str) -> TurnRequest {
    TurnRequest {
        tenant_id: TenantId::new("tenant_a"),
        project_id: ProjectId::from_raw("proj_x"),
        thread_id: ThreadId::new(thread),
        user_text: user_text.into(),
        message_id: None,
        ephemeral_system: None,
    }
}

#[tokio::test]
async fn turn_below_threshold_does_not_compact() {
    let fix = fixture().await;
    // Single low-token turn — nowhere near the 100-token threshold.
    fix.llm.enqueue(assistant_text_with_input_tokens("ok", 10));

    fix.engine
        .handle_turn(turn_request("t1", "hi"))
        .await
        .unwrap();

    let summary = fix
        .db
        .get_thread_summary(&ThreadId::new("t1"))
        .await
        .unwrap();
    assert!(
        summary.is_none(),
        "no compaction should have run; summary present: {summary:?}"
    );
}

#[tokio::test]
async fn turn_above_threshold_triggers_compaction_and_persists_summary() {
    let fix = fixture().await;

    // Drive 3 user turns. Only the third returns input_tokens >=100, so
    // only the third turn fires compaction. The first two keep things
    // conservative so we have a stable history before the trigger fires.
    fix.llm
        .enqueue(assistant_text_with_input_tokens("first", 10));
    fix.engine
        .handle_turn(turn_request("t1", "hello"))
        .await
        .unwrap();
    fix.llm
        .enqueue(assistant_text_with_input_tokens("second", 20));
    fix.engine
        .handle_turn(turn_request("t1", "another"))
        .await
        .unwrap();

    // Third turn: terminal response *and* the summariser reply must be
    // pre-queued in that order.
    fix.llm
        .enqueue(assistant_text_with_input_tokens("third", 250));
    fix.llm
        .enqueue(assistant_text("SUMMARY: user said hi twice"));

    fix.engine
        .handle_turn(turn_request("t1", "trigger"))
        .await
        .unwrap();

    let summary = fix
        .db
        .get_thread_summary(&ThreadId::new("t1"))
        .await
        .unwrap()
        .expect("compaction should have produced a summary");
    assert!(
        summary.summary.contains("SUMMARY"),
        "got: {}",
        summary.summary
    );
    assert!(
        summary.msg_count_before >= 2,
        "summary should fold at least 2 messages, got {}",
        summary.msg_count_before
    );
    assert_eq!(summary.input_tokens_before, 250);
}

#[tokio::test]
async fn load_history_splices_summary_in_place_of_compacted_messages() {
    let fix = fixture().await;

    // Three turns, the third triggers compaction.
    fix.llm.enqueue(assistant_text_with_input_tokens("a", 10));
    fix.engine
        .handle_turn(turn_request("t2", "first"))
        .await
        .unwrap();
    fix.llm.enqueue(assistant_text_with_input_tokens("b", 20));
    fix.engine
        .handle_turn(turn_request("t2", "second"))
        .await
        .unwrap();
    fix.llm.enqueue(assistant_text_with_input_tokens("c", 250));
    fix.llm.enqueue(assistant_text("CONDENSED OLD HISTORY"));
    fix.engine
        .handle_turn(turn_request("t2", "third"))
        .await
        .unwrap();

    // After compaction, the next turn's prompt should contain:
    //   - synthetic [SNACA SUMMARY] preamble
    //   - kept-recent tail (compact_keep_recent=2)
    //   - newly-appended user message ("after")
    fix.llm.enqueue(assistant_text_with_input_tokens("done", 5));
    fix.engine
        .handle_turn(turn_request("t2", "after"))
        .await
        .unwrap();

    // Inspect what the mock saw on its last call. The mock records full
    // requests; pull the tail, which corresponds to the "after" turn.
    let observed = fix.llm.observed_request_count();
    assert!(
        observed >= 4,
        "expected at least 4 LLM calls, got {observed}"
    );

    // Verify history seen by the LLM in the LAST request includes the
    // summary preamble. The mock retains all requests in `requests`
    // — we don't have direct access, but we can re-load history through
    // the engine helper indirectly via the DB: only the messages_after
    // path should be live now, plus the synthetic preamble. Exact text
    // must mention the summariser's output.
    let summary = fix
        .db
        .get_thread_summary(&ThreadId::new("t2"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(summary.summary.trim(), "CONDENSED OLD HISTORY");

    // Live messages after the cutoff: kept tail (2 messages from prior
    // turn 3 = user "third" + assistant "c") + the new "after" pair.
    let live = fix
        .db
        .messages_after(&ThreadId::new("t2"), &summary.summary_until_message_id, 100)
        .await
        .unwrap();
    let texts: Vec<_> = live
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("after")),
        "live tail should include the post-compaction user turn; got {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("done")),
        "live tail should include the post-compaction assistant reply; got {texts:?}"
    );
}

/// Production path: `compact_blocking = false` (the default). The
/// engine fires compaction on a background task and returns the turn
/// immediately. Poll until the summary lands rather than relying on
/// `handle_turn`'s return point.
#[tokio::test]
async fn compaction_runs_in_background_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = Some(100);
    cfg.compact_keep_recent = 2;
    cfg.protect_first_n = 0;
    // compact_blocking left at default (false).
    let engine = Engine::new(llm.clone(), tools, db.clone(), workspace, cfg);

    // Two warm-up turns, then the threshold-crossing third turn.
    llm.enqueue(assistant_text_with_input_tokens("first", 10));
    engine
        .handle_turn(turn_request("t-bg", "hi"))
        .await
        .unwrap();
    llm.enqueue(assistant_text_with_input_tokens("second", 20));
    engine
        .handle_turn(turn_request("t-bg", "again"))
        .await
        .unwrap();
    llm.enqueue(assistant_text_with_input_tokens("third", 250));
    llm.enqueue(assistant_text("ASYNC SUMMARY"));
    engine
        .handle_turn(turn_request("t-bg", "trigger"))
        .await
        .unwrap();

    // Background task should land within a generous wait window.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        if let Some(s) = db.get_thread_summary(&ThreadId::new("t-bg")).await.unwrap() {
            assert!(s.summary.contains("ASYNC SUMMARY"), "got: {}", s.summary);
            return;
        }
        if std::time::Instant::now() >= deadline {
            panic!("background compaction never produced a summary");
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// First-N protection: compaction folds the *middle* and leaves the
/// head messages verbatim. Drives enough turns to exceed
/// `protect_first_n + compact_keep_recent + 2`, fires compaction, then
/// observes the LLM request on the next turn — it should contain the
/// original first user message AND the synthetic preamble, in that
/// order.
#[tokio::test]
async fn protect_first_n_preserves_head_messages() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = Some(100);
    cfg.compact_keep_recent = 2;
    cfg.protect_first_n = 2; // 2 head messages (= 1 turn) protected
    cfg.compact_blocking = true;
    let engine = Engine::new(llm.clone(), tools, db.clone(), workspace, cfg);

    // 5 warm-up turns build a 10-message thread:
    //   T1: U("first goal: build a foo widget")  + A("ack")
    //   T2..T5: regular back-and-forth
    // protect_first_n=2 + compact_keep_recent=2 + middle=2 (>=2) → eligible.
    llm.enqueue(assistant_text_with_input_tokens("ack", 10));
    engine
        .handle_turn(turn_request("t-head", "first goal: build a foo widget"))
        .await
        .unwrap();
    for i in 0..4 {
        llm.enqueue(assistant_text_with_input_tokens(&format!("step {i}"), 20));
        engine
            .handle_turn(turn_request("t-head", &format!("user message {i}")))
            .await
            .unwrap();
    }
    // 6th turn crosses the threshold → compaction fires.
    llm.enqueue(assistant_text_with_input_tokens("triggered", 250));
    llm.enqueue(assistant_text("CONDENSED MIDDLE"));
    engine
        .handle_turn(turn_request("t-head", "long context now"))
        .await
        .unwrap();

    // The persisted summary must mark `summary_from_message_id` —
    // legacy "compress from beginning" would set None.
    let summary = db
        .get_thread_summary(&ThreadId::new("t-head"))
        .await
        .unwrap()
        .unwrap();
    assert!(
        summary.summary_from_message_id.is_some(),
        "with protect_first_n>0 the summary must record where the compressed band starts"
    );

    // Next turn: inspect the LLM request. The history must contain both
    // the original "first goal" user message (preserved head) and the
    // synthetic [SNACA SUMMARY] preamble.
    let before = llm.observed_request_count();
    llm.enqueue(assistant_text_with_input_tokens("ok", 5));
    engine
        .handle_turn(turn_request("t-head", "after compaction"))
        .await
        .unwrap();

    let reqs = llm.observed_requests();
    assert!(reqs.len() > before, "no new LLM call observed");
    let last = reqs.last().unwrap();
    let texts: Vec<String> = last
        .messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
        })
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("first goal")),
        "preserved head should still carry the original first user message; got {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("[SNACA SUMMARY")),
        "synthetic preamble must be present in history; got {texts:?}"
    );
    let head_idx = texts.iter().position(|t| t.contains("first goal")).unwrap();
    let preamble_idx = texts
        .iter()
        .position(|t| t.contains("[SNACA SUMMARY"))
        .unwrap();
    assert!(
        head_idx < preamble_idx,
        "preserved head must precede the SUMMARY preamble (got head={head_idx} preamble={preamble_idx})"
    );
}

/// Append `content` to `thread` as `role`, returning the new row's id.
async fn seed(
    db: &Database,
    thread: &ThreadId,
    role: Role,
    content: Vec<ContentBlock>,
) -> MessageId {
    db.append_message(&NewMessage {
        thread_id: thread.clone(),
        session_id: SessionId::new(),
        role,
        content,
    })
    .await
    .unwrap()
    .id
}

/// A Read call and its image result, seeded as one tool round trip.
async fn seed_image_read(db: &Database, thread: &ThreadId, id: &str) {
    seed(
        db,
        thread,
        Role::Assistant,
        vec![ContentBlock::tool_use(id, "Read", serde_json::json!({}))],
    )
    .await;
    let image = ContentBlock::Image {
        source: ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "A".repeat(1000),
        },
    };
    seed(
        db,
        thread,
        Role::Tool,
        vec![ContentBlock::tool_result(ToolUseId::new(id), vec![image])],
    )
    .await;
}

#[tokio::test]
async fn preserved_head_goes_through_the_image_defences() {
    // Without this: the head kept by `protect_first_n` skipped retirement,
    // the byte cap and collapse, so an image Read in the thread's first
    // turn was re-sent on every request for the rest of its life.
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let thread = ThreadId::new("t-head-image");
    db.insert_thread(&NewThread {
        id: thread.clone(),
        tenant_id: TenantId::new("tenant_a"),
        project_id: ProjectId::from_raw("proj_x"),
    })
    .await
    .unwrap();
    let text = |t: &str| vec![ContentBlock::text(t)];

    // Head: the first turn Reads an image.
    seed(&db, &thread, Role::User, text("look at first.png")).await;
    seed_image_read(&db, &thread, "head").await;
    // Compacted band.
    let from = seed(&db, &thread, Role::User, text("middle")).await;
    let until = seed(&db, &thread, Role::Assistant, text("middle reply")).await;
    // Live: three later turns, an image each — the newest-3 budget.
    for i in 0..3 {
        seed(&db, &thread, Role::User, text(&format!("look at {i}.png"))).await;
        seed_image_read(&db, &thread, &format!("live{i}")).await;
    }
    db.set_thread_summary(&thread, "SUMMARY", &until, Some(&from), 2, 0)
        .await
        .unwrap();

    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = None;
    cfg.protect_first_n = 3;
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let engine = Engine::new(llm.clone(), tools, db, workspace, cfg);
    llm.enqueue(assistant_text("ok"));
    engine
        .handle_turn(turn_request("t-head-image", "thanks"))
        .await
        .unwrap();

    let req = llm.observed_requests().pop().unwrap();
    let results: Vec<(String, bool)> = req
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|b| match b {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => Some((
                tool_use_id.as_str().to_string(),
                content
                    .iter()
                    .any(|c| matches!(c, ContentBlock::Image { .. })),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        results,
        [
            ("head".to_string(), false),
            ("live0".to_string(), true),
            ("live1".to_string(), true),
            ("live2".to_string(), true),
        ],
        "head image retired as the 4th-newest"
    );
}

/// Shrink-retry: when the LLM keeps returning `ContextOverflow`, the
/// engine compacts with progressively tighter `keep_recent` until
/// either a retry succeeds or `compact_max_retries` is exhausted.
#[tokio::test]
async fn context_overflow_retry_shrinks_keep_recent() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = None; // we drive overflow manually
    cfg.compact_keep_recent = 8;
    cfg.protect_first_n = 0; // simpler invariant — focus on shrink behaviour
    cfg.compact_max_retries = 3;
    cfg.compact_blocking = true;
    let engine = Engine::new(llm.clone(), tools, db.clone(), workspace, cfg);

    // Build a long enough thread so compaction has material to fold.
    for i in 0..12 {
        llm.enqueue(assistant_text_with_input_tokens(&format!("reply {i}"), 10));
        engine
            .handle_turn(turn_request("t-shrink", &format!("user {i}")))
            .await
            .unwrap();
    }

    // Now the next turn: 2 ContextOverflows then a success. After each
    // overflow the engine should compact (one summariser reply per
    // compaction) then retry the LLM call.
    //
    // Order in queue (FIFO):
    //   1. ContextOverflow    — main call
    //   2. summariser reply   — compact #1 (keep_recent shrinks 8 → 4)
    //   3. ContextOverflow    — retry
    //   4. summariser reply   — compact #2 (keep_recent 8 → 2)
    //   5. final success      — retry
    llm.enqueue_err(snaca_llm::LlmError::ContextOverflow);
    llm.enqueue(assistant_text("first summary"));
    llm.enqueue_err(snaca_llm::LlmError::ContextOverflow);
    llm.enqueue(assistant_text("second summary"));
    llm.enqueue(assistant_text_with_input_tokens("finally", 5));

    let res = engine
        .handle_turn(turn_request("t-shrink", "trigger overflow"))
        .await;
    assert!(
        res.is_ok(),
        "engine should recover after 2 shrink-retries: {res:?}"
    );

    // Final persisted summary reflects the *last* compaction call (the
    // tighter tail of 2 messages folded a longer middle).
    let summary = db
        .get_thread_summary(&ThreadId::new("t-shrink"))
        .await
        .unwrap()
        .expect("at least one compaction must have run");
    assert_eq!(summary.summary.trim(), "second summary");
}

/// Shrink-retry exhaustion: when every attempt still returns
/// `ContextOverflow`, the error eventually surfaces to the caller
/// rather than looping forever.
#[tokio::test]
async fn context_overflow_retry_gives_up_after_max() {
    let tmp = tempfile::tempdir().unwrap();
    let workspace = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistryBuilder::default().add(EchoTool).build();
    let llm = Arc::new(MockLlmClient::new());
    let mut cfg = EngineConfig::default_for("mock-model");
    cfg.compact_after_input_tokens = None;
    cfg.compact_keep_recent = 4;
    cfg.protect_first_n = 0;
    cfg.compact_max_retries = 2;
    cfg.compact_blocking = true;
    let engine = Engine::new(llm.clone(), tools, db.clone(), workspace, cfg);

    for i in 0..10 {
        llm.enqueue(assistant_text_with_input_tokens(&format!("r{i}"), 10));
        engine
            .handle_turn(turn_request("t-give-up", &format!("u{i}")))
            .await
            .unwrap();
    }

    // 2 compaction-retries permitted; queue 3 overflows interleaved
    // with 2 summariser replies. The third overflow should surface.
    llm.enqueue_err(snaca_llm::LlmError::ContextOverflow);
    llm.enqueue(assistant_text("s1"));
    llm.enqueue_err(snaca_llm::LlmError::ContextOverflow);
    llm.enqueue(assistant_text("s2"));
    llm.enqueue_err(snaca_llm::LlmError::ContextOverflow);

    let res = engine.handle_turn(turn_request("t-give-up", "boom")).await;
    assert!(
        res.is_err(),
        "engine should give up after compact_max_retries"
    );
}

#[tokio::test]
async fn compaction_skipped_when_thread_too_short() {
    let fix = fixture().await;

    // Single turn that exceeds the token threshold — but with only 1 user
    // + 1 assistant message, there's nothing to summarise (we'd be
    // compacting *everything*). The engine should bail.
    fix.llm
        .enqueue(assistant_text_with_input_tokens("only-one", 500));

    fix.engine
        .handle_turn(turn_request("t3", "single"))
        .await
        .unwrap();

    let summary = fix
        .db
        .get_thread_summary(&ThreadId::new("t3"))
        .await
        .unwrap();
    assert!(
        summary.is_none(),
        "single-turn thread shouldn't be compacted; got {summary:?}"
    );
    // Only one LLM call should have happened — the summariser was never invoked.
    assert_eq!(fix.llm.observed_request_count(), 1);
}

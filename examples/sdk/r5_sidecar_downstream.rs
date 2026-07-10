//! Simulates a downstream host (e.g. SciPen Studio's editor mode) driving snaca
//! entirely through the SDK's public surface plus the R5 sidecar metadata API —
//! with zero edits to snaca source.
//!
//! Flow:
//!   1. Own a SQLite store via the re-exported `Database`.
//!   2. Run real DeepSeek turns through an `Agent` that persists into that store.
//!   3. Decorate the thread with an opaque title (`set_thread_meta`).
//!   4. Tag each turn's messages with a downstream `turn_id` (`set_message_meta`).
//!   5. Read the conversation list back via `list_thread_summaries`
//!      (title + turn_count + last_active_at) and the per-message turn tags.
//!
//! Run: DEEPSEEK_API_KEY=... cargo run -p snaca-sdk --example r5_sidecar_downstream

use snaca_sdk::{Database, ProjectId, TenantId, ThreadId};

#[tokio::main]
async fn main() -> snaca_sdk::Result<()> {
    std::fs::create_dir_all("./data-r5-demo").ok();

    // (1) A downstream owns its own SQLite store via the re-exported Database.
    //     `open` runs migrations, which create the R5 sidecar tables.
    let db = Database::open("./data-r5-demo/state.db").await?;

    let tenant = TenantId::new("scipen");
    let project = ProjectId::from_raw("editor");
    let thread = ThreadId::new("conv-1");

    // (2) An agent that persists into the downstream's own db.
    let agent = snaca_sdk::AgentBuilder::new()
        .llm(snaca_sdk::llm::deepseek_from_env("deepseek-v4-pro")?)
        .read_only_agent_defaults()
        .store(db.clone())
        .tenant_id(tenant.clone())
        .project_id(project.clone())
        .thread_id(thread.clone())
        .data_root("./data-r5-demo")
        .build()
        .await?;

    // --- turn 1 ---
    let out1 = agent.run("用一句话介绍你自己").await?;
    println!("assistant #1: {}", out1.text);
    tag_untagged_messages(&db, &thread, "turn-1").await?;

    // (3) Downstream decorates the thread with its own opaque metadata.
    db.set_thread_meta(&thread, &serde_json::json!({ "title": "关于自我介绍" }))
        .await?;

    // --- turn 2 ---
    let out2 = agent.run("再用一句话补充一个冷知识").await?;
    println!("assistant #2: {}", out2.text);
    tag_untagged_messages(&db, &thread, "turn-2").await?;

    // (5a) Conversation list — title + turn_count + activity, one query.
    println!("\n=== list_thread_summaries ===");
    for s in db.list_thread_summaries(&tenant, &project).await? {
        let title = s
            .meta
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("(untitled)");
        println!(
            "thread={} title={:?} messages={} turns={} last_active={:?}",
            s.thread.id.as_str(),
            title,
            s.message_count,
            s.turn_count,
            s.last_active_at,
        );
    }

    // (5b) Per-message turn tags, batch-read to avoid N+1.
    println!("\n=== get_message_meta_for_thread ===");
    let mut metas = db.get_message_meta_for_thread(&thread).await?;
    metas.sort_by_key(|(id, _)| id.to_string());
    for (id, data) in metas {
        println!(
            "message={} turn_id={:?}",
            id,
            data.get("turn_id").and_then(|t| t.as_str()).unwrap_or("?")
        );
    }

    Ok(())
}

/// Stamp `turn_id` onto every message in the thread that isn't tagged yet.
/// Uses only re-exported Database methods — no snaca source dependency.
async fn tag_untagged_messages(
    db: &Database,
    thread: &ThreadId,
    turn_id: &str,
) -> snaca_sdk::Result<()> {
    for m in db.recent_messages(thread, 100).await? {
        if db.get_message_meta(&m.id).await?.is_none() {
            db.set_message_meta(&m.id, &serde_json::json!({ "turn_id": turn_id }))
                .await?;
        }
    }
    Ok(())
}

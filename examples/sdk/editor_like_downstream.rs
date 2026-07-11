//! Full-feature downstream simulation (0.3.0 acceptance harness).
//!
//! Stands up an "editor-like" host that drives snaca using ONLY the
//! `snaca_sdk` public facade — no snaca-internal crate is imported. Exercises
//! every extension point the submodule integration depends on:
//!
//!   R7  custom `LlmClient` injected (here a scripted mock — a downstream
//!       OpenAI-compatible provider plugs in the exact same way)
//!   R4  explicit tool cwd decoupled from the SNACA metadata/data root
//!   R6  standard tool set + a downstream `Tool` appended
//!   R2  a downstream `HostContext` (editor reverse-RPC) …
//!   R3  … reached by that tool via `ctx.host_context()`
//!   R1  per-turn `ephemeral_system` (open file / selection)
//!   R5  thread/message sidecar metadata + conversation summaries
//!   M1  skills / mcp / memory subsystems assembled from the facade
//!
//! Because it imports only `snaca_sdk`, this file doubles as the machine-checked
//! proof that a zero-source-diff submodule integration is possible. The
//! `check-sdk-boundaries.sh` gate enforces the facade-only import rule.
//!
//! Run: cargo run -p snaca-sdk --example editor_like_downstream

use async_trait::async_trait;
use serde_json::{json, Value};
use snaca_sdk::mcp::McpServerConfig;
use snaca_sdk::memory::MemoryStore;
use snaca_sdk::skills::{Skill, SkillRegistry, SkillScope};
use snaca_sdk::{
    AgentBuilder, AgentInput, ApprovalRequirement, ContentBlock, Database, HostContext,
    HostContextError, LlmClientTrait, Message, MessageRequest, MessageResponse, ProjectId,
    ProviderCaps, Role, StopReason, TenantId, ThreadId, Tool, ToolCapabilities, ToolContext,
    ToolError, ToolOutput, ToolResult, ToolUseId, Usage,
};
use std::sync::Arc;

// --- R7: a downstream-defined LLM provider (scripted for determinism) ---------
struct ScriptedLlm;

#[async_trait]
impl LlmClientTrait for ScriptedLlm {
    fn provider_name(&self) -> &'static str {
        "scripted"
    }
    fn model(&self) -> &str {
        "scripted-model"
    }
    fn capabilities(&self) -> ProviderCaps {
        ProviderCaps {
            tool_use: true,
            ..Default::default()
        }
    }
    async fn create_message(&self, req: MessageRequest) -> snaca_sdk::LlmResult<MessageResponse> {
        let saw_tool_result = req.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
        });
        let (content, stop_reason) = if saw_tool_result {
            (vec![ContentBlock::text("done")], StopReason::EndTurn)
        } else {
            (
                vec![ContentBlock::ToolUse {
                    id: ToolUseId::new("call_1"),
                    name: "zotero_search".into(),
                    input: json!({"query": "rust ownership"}),
                }],
                StopReason::ToolUse,
            )
        };
        Ok(MessageResponse {
            id: "msg_1".into(),
            message: Message::new(Role::Assistant, content),
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                ..Default::default()
            },
            stop_reason,
        })
    }
}

// --- R2: the editor host's reverse-RPC endpoint -------------------------------
#[derive(Debug)]
struct EditorHost;

#[async_trait]
impl HostContext for EditorHost {
    async fn call(&self, method: &str, params: Value) -> Result<Value, HostContextError> {
        match method {
            "zotero.search" => Ok(json!({
                "hits": [{"key": "ABC123", "title": format!("Result for {}", params["query"])}]
            })),
            other => Err(HostContextError::HostRejected(format!(
                "unknown method {other}"
            ))),
        }
    }
}

// --- R6 + R3: a downstream tool that reverse-RPCs the host --------------------
#[derive(Debug)]
struct ZoteroSearchTool;

#[async_trait]
impl Tool for ZoteroSearchTool {
    fn name(&self) -> &str {
        "zotero_search"
    }
    fn description(&self) -> &str {
        "Search the user's Zotero library via the editor host."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object", "properties": {"query": {"type": "string"}}})
    }
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::Never
    }
    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let host = ctx
            .host_context()
            .ok_or_else(|| ToolError::Other("no editor host attached".into()))?;
        let resp = host
            .call("zotero.search", input)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput::text(resp.to_string()))
    }
}

#[tokio::main]
async fn main() -> snaca_sdk::Result<()> {
    let base = std::env::temp_dir().join("snaca-editor-demo");
    // Start from a clean slate so re-runs are deterministic — otherwise the
    // persistent sqlite db under `base` accumulates messages/turns across runs.
    std::fs::remove_dir_all(&base).ok();
    let data_root = base.join("data");
    let project_dir = base.join("user-project");
    std::fs::create_dir_all(&data_root).ok();
    std::fs::create_dir_all(&project_dir).ok();

    // A downstream owns its store via the facade `Database`.
    let db = Database::open(data_root.join("state.db")).await?;

    let tenant = TenantId::new("scipen");
    let project = ProjectId::from_raw("editor");
    let thread = ThreadId::new("conv-1");

    // R6: standard tool set + the downstream Zotero tool.
    let tools = snaca_sdk::tools::base_tool_registry_builder()
        .add(ZoteroSearchTool)
        .build();

    let agent = AgentBuilder::new()
        .llm(ScriptedLlm) // R7
        .tools(tools) // R6
        .store(db.clone())
        .explicit_workspace(&project_dir) // R4: cwd = real project, metadata = data_root
        .data_root(&data_root)
        .tenant_id(tenant.clone())
        .project_id(project.clone())
        .thread_id(thread.clone())
        // R2/R3: mint a fresh host handle per turn, keyed on the turn id.
        .host_context_factory(Arc::new(|_turn_id| {
            Arc::new(EditorHost) as Arc<dyn HostContext>
        }))
        .build()
        .await?;

    // R1: inject volatile editor context for this turn.
    let out = agent
        .run(
            AgentInput::new("find sources on rust ownership")
                .ephemeral_system("## Editor context\nopen file: src/main.rs\nselection: fn main"),
        )
        .await?;
    println!("assistant: {}", out.text);

    // R5: decorate the conversation, tag turns, read the summary back.
    db.set_thread_meta(&thread, &json!({"title": "Rust ownership sources"}))
        .await?;
    for m in db.recent_messages(&thread, 100).await? {
        db.set_message_meta(&m.id, &json!({"turn_id": "turn-1"}))
            .await?;
    }
    for s in db.list_thread_summaries(&tenant, &project).await? {
        let title = s
            .meta
            .as_ref()
            .and_then(|m| m.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("(untitled)");
        println!(
            "thread={} title={:?} messages={} turns={}",
            s.thread.id.as_str(),
            title,
            s.message_count,
            s.turn_count
        );
    }

    // M1: skills / mcp / memory all assembled from the facade alone.
    let skill = Skill::from_str(
        "---\nname: reviewer\ndescription: careful reviewer\nwhen_to_use: reviews\n---\nTreat every file like prod.\n",
        SkillScope::Tenant,
        None,
    )
    .expect("valid skill");
    let skills = SkillRegistry::from_skills(vec![skill]);
    println!("skills assembled: {} registered", skills.len());

    let mem = MemoryStore::new(data_root.join("mem"));
    println!("memory store rooted at {:?}", mem.root());

    let mcp: McpServerConfig = serde_json::from_value(json!({
        "name": "fs",
        "command": "echo",
        "args": ["hello"],
    }))
    .expect("valid mcp config");
    println!("mcp server config reachable: name={}", mcp.name);

    println!("\nAll extension points exercised via snaca_sdk facade only.");
    Ok(())
}

//! End-to-end check that a per-turn `HostContext` factory (R2) is injected
//! into the tool context and reachable by a tool via `ctx.host_context()`.
//!
//! Scripts the mock LLM to call a custom tool that reverse-RPCs the host;
//! asserts the host received the call (with the turn id and opaque params) and
//! its response flowed back to the model.

use async_trait::async_trait;
use serde_json::{json, Value};
use snaca_core::{ProjectId, TenantId, ThreadId};
use snaca_engine::{Engine, EngineConfig, TurnRequest};
use snaca_state::Database;
use snaca_tools_api::{
    ApprovalRequirement, HostContext, HostContextError, Tool, ToolCapabilities, ToolContext,
    ToolError, ToolOutput, ToolRegistry, ToolResult,
};
use snaca_workspace::WorkspaceLayout;
use std::sync::{Arc, Mutex};

mod common;
use common::{assistant_text, assistant_tool_call, MockLlmClient};

/// Records every reverse-RPC and answers with a canned hit.
#[derive(Debug)]
struct RecordingHost {
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

#[async_trait]
impl HostContext for RecordingHost {
    async fn call(&self, method: &str, params: Value) -> Result<Value, HostContextError> {
        self.calls
            .lock()
            .unwrap()
            .push((method.to_string(), params));
        Ok(json!({ "hit": "found-1" }))
    }
}

/// Tool that asks the host for data and returns the host's `hit`.
#[derive(Debug)]
struct HostPingTool;

#[async_trait]
impl Tool for HostPingTool {
    fn name(&self) -> &str {
        "host_ping"
    }
    fn description(&self) -> &str {
        "Ask the host for a bibliography hit."
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }
    fn approval_requirement(&self) -> ApprovalRequirement {
        ApprovalRequirement::Never
    }
    async fn execute(&self, _input: Value, ctx: &ToolContext) -> ToolResult {
        let host = ctx
            .host_context()
            .ok_or_else(|| ToolError::Other("no host context attached".into()))?;
        let resp = host
            .call("zotero.search", json!({"q": "rust"}))
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let hit = resp
            .get("hit")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        Ok(ToolOutput::text(hit))
    }
}

#[tokio::test]
async fn host_context_factory_is_injected_and_reachable_by_tools() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistry::builder().add(HostPingTool).build();

    let llm = Arc::new(MockLlmClient::new());
    llm.enqueue(assistant_tool_call(vec![(
        "call_1",
        "host_ping",
        json!({}),
    )]));
    llm.enqueue(assistant_text("done"));

    let calls = Arc::new(Mutex::new(Vec::new()));
    let calls_for_factory = calls.clone();
    let seen_turn_ids = Arc::new(Mutex::new(Vec::new()));
    let seen_for_factory = seen_turn_ids.clone();

    let engine = Engine::new(
        llm,
        tools,
        db.clone(),
        layout,
        EngineConfig::default_for("mock-model"),
    )
    .with_host_context_factory(Arc::new(move |turn_id: String| {
        seen_for_factory.lock().unwrap().push(turn_id);
        Arc::new(RecordingHost {
            calls: calls_for_factory.clone(),
        }) as Arc<dyn HostContext>
    }));

    let outcome = engine
        .handle_turn(TurnRequest {
            tenant_id: TenantId::new("t"),
            project_id: ProjectId::from_raw("p"),
            thread_id: ThreadId::new("chat_host"),
            user_text: "look it up".into(),
            message_id: Some("msg-42".into()),
            ephemeral_system: None,
        })
        .await
        .unwrap();

    assert_eq!(outcome.iterations, 2);
    // The factory was called once, keyed on the turn id (the IM message id).
    assert_eq!(seen_turn_ids.lock().unwrap().as_slice(), &["msg-42"]);
    // The tool reached the host with the opaque method + params.
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "zotero.search");
    assert_eq!(calls[0].1, json!({"q": "rust"}));
}

#[tokio::test]
async fn no_factory_means_tools_see_no_host_context() {
    let tmp = tempfile::tempdir().unwrap();
    let layout = WorkspaceLayout::new(tmp.path()).unwrap();
    let db = Database::open_in_memory().await.unwrap();
    let tools = ToolRegistry::builder().add(HostPingTool).build();

    let llm = Arc::new(MockLlmClient::new());
    llm.enqueue(assistant_tool_call(vec![(
        "call_1",
        "host_ping",
        json!({}),
    )]));
    llm.enqueue(assistant_text("done"));

    // No factory attached → host_context() is None → the tool errors cleanly,
    // and the turn still completes (the error surfaces as a tool result).
    let engine = Engine::new(
        llm,
        tools,
        db.clone(),
        layout,
        EngineConfig::default_for("mock-model"),
    );

    let outcome = engine
        .handle_turn(TurnRequest {
            tenant_id: TenantId::new("t"),
            project_id: ProjectId::from_raw("p"),
            thread_id: ThreadId::new("chat_host_none"),
            user_text: "look it up".into(),
            message_id: None,
            ephemeral_system: None,
        })
        .await
        .unwrap();
    assert_eq!(outcome.assistant_text, "done");
}

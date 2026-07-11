//! Executable snapshot of the `snaca-sdk` semver-stable public surface.
//!
//! Everything a downstream integration (SciPen Studio) depends on to run as a
//! zero-source-diff submodule must remain reachable through `snaca_sdk`. If a
//! symbol below is removed or renamed, this test stops compiling — a loud,
//! intentional signal that the stable facade changed and the CHANGELOG /
//! semver bump must reflect it.

#![allow(unused_imports)]

// Core value types.
use snaca_sdk::{
    ContentBlock, Message, MessageId, ProjectId, Role, SessionId, TenantId, ThreadId, ToolSchema,
    ToolUseId, Usage,
};

// Agent facade + builders.
use snaca_sdk::{
    Agent, AgentBuilder, AgentInput, AgentOutput, EngineRuntimeBuilder, Result, SdkError,
};

// Engine surface (M1 uplift).
use snaca_sdk::{
    Engine, EngineConfig, HostContextFactory, RuntimeToolFactory, TurnEventListener, TurnOutcome,
};

// LLM provider surface (R7).
use snaca_sdk::{
    LlmClientTrait, LlmError, LlmResult, MessageRequest, MessageResponse, ProviderCaps,
    RetryConfig, RetryingLlmClient, StopReason, StreamEvent,
};

// Tool + host reverse-RPC surface (R2/R3/R6).
use snaca_sdk::{
    ApprovalRequirement, HostContext, HostContextError, Tool, ToolCapabilities, ToolContext,
    ToolError, ToolOutput, ToolRegistry, ToolRegistryBuilder, ToolResult,
};

// State surface incl. R5 sidecar row + write inputs (M1 uplift).
use snaca_sdk::{
    Database, MessageRow, NewMessage, NewThread, StateError, StateResult, ThreadRow,
    ThreadSummaryRow,
};

// Workspace surface (R4 uplift).
use snaca_sdk::{WorkspaceError, WorkspaceLayout};

// Approval / question / memory-provider contracts.
use snaca_sdk::{ApprovalGate, ConversationStore, MemoryProvider, NoopApprovalGate, QuestionGate};

// Subsystem modules (M1 uplift): skills, narrowed mcp, memory store side.
use snaca_sdk::mcp::{McpManager, McpServerConfig, McpTransport};
use snaca_sdk::memory::{MemoryError, MemoryScope, MemoryStore};
use snaca_sdk::skills::{Skill, SkillProvider, SkillRegistry, SkillRegistryBuilder, SkillScope};

// Composable standard tool-registry builders (R6).
use snaca_sdk::tools::{base_tool_registry_builder, read_only_registry_builder};

#[test]
fn stable_facade_symbols_are_reachable() {
    // Compilation of the imports above IS the assertion. A trivial runtime
    // touch keeps the test from being optimized into nothing.
    let _ = ProjectId::from_raw("p");
}

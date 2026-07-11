//! MCP server management for SDK users (M1 facade uplift, narrowed).
//!
//! Re-exports the *rmcp-free* subset of `snaca-mcp`: a host can configure and
//! own MCP servers via [`McpManager`] and get their tools with
//! `McpManager::tools_for(...)`. The lower-level `McpClient` / `McpTool` are
//! deliberately NOT re-exported — their public signatures leak `rmcp` types,
//! and pinning the SDK's semver surface to an external SDK is a non-goal. A
//! host that needs that depth should depend on `snaca-mcp` directly.
pub use snaca_mcp::{McpManager, McpServerConfig, McpTransport};

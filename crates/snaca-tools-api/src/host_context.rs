//! Domain-agnostic reverse-RPC from a tool to its embedding host (R2).
//!
//! Some tools need to call *back* into the host mid-execution to fetch data or
//! trigger a host action (e.g. an editor host serving file contents, prompting
//! the user, or a bibliography lookup). snaca stays entirely unaware of what
//! those methods mean: `method` is an opaque string the host and the tool agree
//! on (e.g. `"zotero.search"`, `"editor.file_content"`), and `params` / the
//! return value are opaque JSON. The transport (stdio duplex, IPC, in-process)
//! lives entirely in the host — snaca only defines this trait and the injection
//! point ([`ToolContext::host_context`](crate::ToolContext::host_context) plus
//! the engine's per-turn factory).

use async_trait::async_trait;
use serde_json::Value;

/// A handle a tool uses to call back into the embedding host. Domain-agnostic
/// on purpose: snaca never interprets `method` or the payloads.
#[async_trait]
pub trait HostContext: Send + Sync + std::fmt::Debug {
    /// Invoke a host-defined `method` with opaque JSON `params`, returning the
    /// host's opaque JSON response. Both sides of the contract are owned by the
    /// host and its tools; snaca just relays.
    async fn call(&self, method: &str, params: Value) -> Result<Value, HostContextError>;
}

/// Failure modes surfaced to a tool when a host reverse-RPC does not succeed.
/// `#[non_exhaustive]` so new variants can be added without a breaking change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HostContextError {
    /// The host received the request but declined it (permission, unknown
    /// method, bad arguments, …). The string is host-supplied detail.
    #[error("host rejected request: {0}")]
    HostRejected(String),
    /// The host did not respond within the tool/host-agreed deadline.
    #[error("timed out")]
    Timeout,
    /// No host context is reachable (no transport, host went away, or no
    /// factory was injected into the engine).
    #[error("host context unavailable: {0}")]
    Unavailable(String),
    /// The host responded, but the payload could not be understood.
    #[error("invalid response payload: {0}")]
    InvalidPayload(String),
}

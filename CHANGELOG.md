# Changelog

All notable changes to SNACA are documented here.

## Stable API boundary

`snaca-sdk` (the crate root plus its `mcp`, `memory`, `skills`, `tools`,
`workspace`, `store`, `config`, `llm` modules and everything they `pub use`) is
the **semver-stable public surface**. A downstream host can integrate snaca as a
git submodule depending on `snaca-sdk` alone. The internal library crates
(`snaca-engine`, `snaca-state`, `snaca-tools`, …) beyond what the facade
re-exports are **not** covered by semver and may change between minor versions.

## 0.3.0

Downstream extension points — a host (e.g. an editor sidecar) can now drive
snaca entirely through the `snaca-sdk` facade with **zero source edits**. All
changes are additive with neutral defaults; existing IM-mode deployments are
unaffected.

### Added
- **R1 — per-turn ephemeral system context.** `TurnRequest.ephemeral_system` /
  `AgentInput::ephemeral_system` inject a volatile system fragment after the
  cacheable prefix (open file, selection, …) without busting the prompt cache.
  `None` is byte-identical to before.
- **R2/R3 — domain-agnostic host reverse-RPC.** `HostContext`
  (`call(method, params) -> Value`) and `HostContextError` (`#[non_exhaustive]`);
  `ToolContext::with_host_context` / `host_context()`; the engine's
  `HostContextFactory` and `Engine::with_host_context_factory` /
  `AgentBuilder::host_context_factory`. Method names, payloads, and transport
  are the host's; snaca relays opaque JSON.
- **R4 — explicit tool cwd.** `WorkspaceLayout::with_explicit_workspace` /
  `AgentBuilder::explicit_workspace` pin the tool cwd to the user's real project
  while metadata (memory/skills/db) stays under `data_root`.
- **R5 — downstream-writable sidecar metadata** (shipped in 0.2.8): `thread_meta`
  / `message_meta`, `Database::{set,get}_thread_meta` / `{set,get}_message_meta`
  / `get_message_meta_for_thread` / `list_thread_summaries` → `ThreadSummaryRow`.
- **R6 — composable tool registry.** `base_tool_registry_builder()` /
  `read_only_registry_builder()` are public; re-exported `ToolRegistryBuilder`.
  Downstream: `base_tool_registry_builder().add(CustomTool).build()`.
- **R7 — provider injection.** The `LlmClient` / `ProviderCaps` /
  `MessageRequest` / `MessageResponse` / `RetryingLlmClient` surface is committed
  as stable; a host injects any `Arc<dyn LlmClient>` via `llm_arc`.
- **M1 — facade uplift.** `snaca-sdk` now re-exports `Engine`, `EngineConfig`,
  `NoopApprovalGate`, `TurnEventListener`, `NewThread`, `NewMessage`,
  `WorkspaceLayout`, `WorkspaceError`; new `snaca_sdk::skills` module; new
  `snaca_sdk::mcp` module (narrowed to `McpManager` + `McpServerConfig` +
  `McpTransport` — the rmcp-free subset); `snaca_sdk::memory` gains the store
  side (`MemoryStore`, `MemoryScope`, `MemoryError`).

### Changed
- `snaca_memory::MemoryError` is now `#[non_exhaustive]`.
- `ToolContext`'s internal state derives `Clone`; `with_*` setters are
  clone-mutate (no external behavior change).

### Notes
- `examples/sdk/editor_like_downstream.rs` exercises R1–R7 + skills/mcp/memory
  through the facade alone; `scripts/check-sdk-boundaries.sh` enforces that the
  downstream harnesses import no snaca-internal crate.

## 0.2.8
- **R5** — downstream-writable sidecar metadata (`thread_meta` / `message_meta`)
  and conversation summaries. See above.

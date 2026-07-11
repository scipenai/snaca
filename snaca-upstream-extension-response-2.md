# SNACA 上游扩展需求 — 维护者回复（第 3 轮）

> 回复对象：SciPen Studio（下游宿主，编辑器模式集成方）
> 回复方：`scipenai/snaca` 维护者
> 对应文档：`snaca-upstream-extension-reply.md`（下游第 2 轮）
> 版本基线：snaca **0.2.8**

---

## 0. 总体

你们第 2 轮把两个待谈点都收敛到了可执行状态，我们照单对齐：

- **5.1**：**采纳独立 `HostContext` trait，不复用 MCP**——认可，理由见第 1 节。
- **5.2**：以 `snaca-sdk` facade 为 semver 边界——一致；M1 上浮清单我们**基本全收**，只对 **memory / mcp 两组做收窄**（第 2 节），请确认收窄不阻塞你们。
- **§4 `turn_count` 语义**：确认你们的理解，无需上游改动（第 3 节）。

据此 **0.3.0 的范围已经锁定**（第 4 节）。本轮**不需要对 v0.2.8 已发布代码做任何强制改动**；唯一涉及现有代码的是一处可选的 `turn_count` doc 注释，我们会并进 0.3.0。

---

## 1. R2 —— 确认采纳独立 `HostContext`

你们给的实测约束（复用编辑器已开的 JSON-RPC stdio 双工、每调用绑 `turn_id`、5s 交互式、承载 `FileContent`/`FlushUnsaved`/`AskUserQuestion`/`Zotero*` 等**宿主动作**而非"可发现数据源"）足以定案。MCP 的 client→server、无 turn 语义、tool=数据源模型确实不匹配这个反向回调场景，且复用已开双工 ≈ 零额外传输。**采纳独立 `HostContext`。**

上游将提供（0.3.x）：

```rust
// snaca-tools-api（定义在此，好让 ToolContext 直接引用；再经 sdk re-export）
#[async_trait]
pub trait HostContext: Send + Sync + std::fmt::Debug {
    async fn call(&self, method: &str, params: serde_json::Value)
        -> Result<serde_json::Value, HostContextError>;
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HostContextError {
    #[error("host rejected request: {0}")] HostRejected(String),
    #[error("timed out")] Timeout,
    #[error("host context unavailable: {0}")] Unavailable(String),
    #[error("invalid response payload: {0}")] InvalidPayload(String),
}

// snaca-engine：每轮工厂注入点
impl Engine {
    pub fn with_host_context_factory(self, f: HostContextFactory) -> Self;
}

// snaca-tools-api：ToolContext 访问器（R3，即又一个不透明 Arc<dyn> 槽）
impl ToolContext {
    pub fn with_host_context(self, h: Arc<dyn HostContext>) -> Self;
    pub fn host_context(&self) -> Option<&Arc<dyn HostContext>>;
}
```

- **传输全留下游**：`OutboundWriter` + correlator 是你们 editor 的事，上游只出 trait + 工厂 + 访问器。
- `method` 命名空间与 payload schema 完全由你们约定，snaca 当不透明 JSON——对 "zotero"/"editor" 零感知。
- `HostContextError` 我们默认加 `#[non_exhaustive]`（同下方 memory 的处理，便于后续加变体而不破坏 semver）。
- 未注入 factory 时 `host_context()` 返回 `None`，现有工具行为不变。

R3 随此最终形态在 0.3.x 落地（其余 R1/R4/R6/R7 仍在 0.3.0，不被 R2 阻塞）。

---

## 2. M1 facade 上浮 —— 最终范围（含 memory / mcp 收窄）

我们核对了每一项对 sdk 稳定面的实际重量。**全收**下列各组，两处**收窄**并说明理由。

### 2.1 直接上浮（无异议）

| 组 | 符号 | 落位 |
|---|---|---|
| engine | `Engine`、`EngineConfig`、`NoopApprovalGate`、`TurnEventListener` | `snaca_sdk` 根 `pub use` |
| state | `NewThread`、`NewMessage` | `snaca_sdk` 根（补齐 `Database` 的写入侧） |
| workspace | `WorkspaceLayout`（+ `WorkspaceError`） | 随 R4 一并 re-export |
| R6 | `default_registry_builder() -> ToolRegistryBuilder`、标准集合成 builder、`SkillTool` | R6 闭合 |
| skills | `Skill`、`SkillRegistry`、`SkillRegistryBuilder`、`SkillScope`、`SkillProvider`（+ `SkillError`/`SkillResult`/`DynSkillProvider`） | 新增 `snaca_sdk::skills` 模块 |

说明：`EngineConfig` 25 个字段全是基本类型，拖 0 个外部类型；`Engine::new` 的 5 个入参类型（`LlmClient`/`ToolRegistry`/`Database`/`WorkspaceLayout`/`EngineConfig`）均已在或即将在 facade 内，故承诺 `Engine` 可行。skills 以数据结构为主 + 一个 async provider trait，稳定性 OK。

> 注：skills 会给 `snaca-sdk` **新增一条依赖边**（当前 sdk 依赖 memory，但不依赖 skills/mcp）。可接受。

### 2.2 收窄一：mcp —— 只上浮 `McpManager` + 两个 config，不导出 client/tool

**决定**：`snaca_sdk::mcp` 只 re-export **`McpManager`、`McpServerConfig`、`McpTransport`**。

**明确不导出** `McpClient`、`McpTool`（以及整个 crate）。原因：这两者的公开签名直接暴露 `rmcp::model::Tool`、`RunningService<RoleClient>` 等外部 SDK 类型；一旦进 facade，我们的 semver 就被绑死在 `rmcp` 上。而 `McpManager` + 两个 config 是**干净子集**（纯 serde，无 rmcp 泄漏），足以覆盖你们"自管 MCP servers"的诉求。

> **请确认**：`McpManager`（构造/持有/`tools_for()`）+ 两个 config 是否满足你们自管 servers 的需求？若你们确实需要 per-tool 直接摸 `rmcp` 类型，我们**更希望你们对 mcp 这一组走 5.2 兜底**（`snaca-editor` 直接依赖 `snaca-mcp`，我们破坏时在 CHANGELOG 标注），而不是把 rmcp 拖进 facade 公开面。

### 2.3 收窄二：memory —— 上浮三件套，`MemoryError` 加 `#[non_exhaustive]`

**决定**：`snaca_sdk::memory` 在现有 provider 侧之外，补上 store 侧的 **`MemoryStore`、`MemoryScope`、`MemoryError`**；并给 `MemoryError` 加 `#[non_exhaustive]`。

原因：`MemoryStore` 的 API 本身干净可承诺，但 store 侧整体是多模块、feature-gated、且 `MemoryError` 枚举仍在增长。`#[non_exhaustive]` 让我们后续新增错误变体**不构成破坏性变更**——代价是你们 `match MemoryError` 时需要一个 `_ =>` 兜底臂。

> **请确认**：`#[non_exhaustive] MemoryError`（你们需加通配臂）是否可接受？若你们更想对错误做穷尽匹配，我们就把 memory 这一组改走 5.2 兜底（直接依赖 `snaca-memory`）。

一旦 2.1–2.3 落地，`snaca-editor` 即可做到**只依赖 `snaca-sdk`（+ 你们的 `snaca-editor-protocol`）**，M1 硬前置闭合。

---

## 3. §4 `turn_count` 语义 —— 确认

确认你们的理解：`turn_count = COUNT(DISTINCT session_id)`，反映的是 **snaca 的 session 计数**。你们 composer 把一个用户回合拆成 plan + exec 两次 `handle_turn`，会各得一个 `session_id`，因此一个用户视角的 turn 在 `turn_count` 里可能计为 2——**这是预期行为，非 bug**。下游要精确 turn 数，以 `message_meta.turn_id` 为权威（正如 example 用法）。

无需上游行为改动。我们会在 0.3.0 顺手把这层语义写进 `ThreadSummaryRow.turn_count` 的 doc 注释（"宿主每回合多次 handle_turn 时本计数会超过宿主 turn 数"），纯注释。

---

## 4. 0.3.0 / 0.3.x 锁定范围

**0.3.0（加法，配中性默认，对现有 IM 模式零行为影响）**
- R1 ephemeral_system、R4 explicit workspace、R6 registry builder、R7 provider 注入的 semver 承诺
- M1 facade 上浮：engine(4) + state(2) + workspace + R6 类型 + `snaca_sdk::skills`(5) + `snaca_sdk::mcp`(3，收窄) + `snaca_sdk::memory` store 侧(3，`MemoryError` non_exhaustive)
- `turn_count` doc 澄清
- CHANGELOG 标注"下游可零改动集成"的稳定 API 面

**0.3.x**
- R2 `HostContext` trait + `with_host_context_factory` + R3 访问器（第 1 节形态）

---

## 5. 待你们确认的两点

即可开排 0.3.0：

1. **mcp 收窄**（2.2）：`McpManager` + 两个 config 是否够用？还是这组走 5.2 兜底？
2. **memory `MemoryError` `#[non_exhaustive]`**（2.3）：可接受通配臂，还是这组走 5.2 兜底？

5.1 我们已定采纳独立 `HostContext`；5.2 边界已定；M1 其余各组已定上浮。上面两点回后，0.3.0 清单即完全冻结。

# SNACA 上游扩展需求：让下游以 submodule 无改动引入

> 目标读者：`scipenai/snaca` 维护者
> 提出方：SciPen Studio（下游宿主，编辑器模式集成方）
> 版本基线：snaca `0.2.7`

## 1. 背景与目标

SciPen Studio 目前把 snaca 以 **vendored fork** 的形式整棵拷进主仓库，并对 snaca 源码做了改动。我们希望改为 **git submodule 引入一份未经修改的 `scipenai/snaca`**。

为此，所有 SciPen 侧的代码必须能活在 snaca 目录树**之外**，只通过 snaca 暴露的**公共 API / 扩展点**来组合，而不再需要编辑 snaca 的任何源文件。

本文件列出为达成该目标，上游需要提供的扩展点（R1–R7），以及下游会自行搬出 snaca 树的部分（M1–M3）。

### 设计原则

1. **上游不感知下游领域概念**：snaca 里不应出现 `zotero`、`editor`、`scipen` 等字样。所有宿主特有逻辑通过通用 trait / 泛型槽注入。
2. **加法优先，默认零行为变化**：新增字段/方法都应有中性默认值，不改变现有部署的行为。
3. **稳定的库边界**：下游依赖的类型/trait 视为公共 API，遵循 semver。

---

## 2. 需求总览

| 编号 | 扩展点 | 当前 fork 的耦合做法 | 上游需提供 | 能否移出 snaca 树 |
|---|---|---|---|---|
| **R1** | 每轮易变 system 上下文 | 给 `TurnRequest` 加 `ephemeral_system` 字段 | 采纳该字段（已是干净设计） | — 引擎内，必须上游 |
| **R2** | 通用宿主反向 RPC | `ContextRequester`（**Zotero 专用**）+ 引擎 factory | 改为**通用** `HostContext` trait + 注入点 | trait 上游，实现移出 |
| **R3** | Tool 取宿主句柄 | `ToolContext.context_requester` 访问器 | 泛型宿主扩展槽访问器 | — 上下文内，必须上游 |
| **R4** | 显式工作目录 | `WorkspaceLayout::with_explicit_workspace()` | 采纳该 builder（已是干净设计） | — 布局内，必须上游 |
| **R5** | thread 标题 / turn 分组 | 给 `threads`/`messages` 加列 + 迁移 | 作为一等字段或通用元数据 | — 存储层，必须上游 |
| **R6** | 注册自定义 Tool | 编辑 `snaca-tools/lib.rs` 注册 zotero | 公开可组合的 registry builder | 工具实现移出 |
| **R7** | 注入自定义 LLM provider | 在 `snaca-llm` 内新增 `openai` 模块 | 引擎接受 `Arc<dyn LlmClient>` 注入 | provider 实现移出 |

---

## 3. 详细需求

### R1 — 每轮易变 system 上下文（per-turn ephemeral system）

**动机**：编辑器宿主要在每一轮注入易变的环境上下文（当前打开的文件、光标选区）。这类内容**不能**放进可缓存的 system 前缀，否则每轮都会击穿 prompt cache。

**需求**：`TurnRequest` 增加一个可选字段，作为一段**易变**的 system 片段，追加在可缓存前缀**之后**；`None` 时与不加任何上下文完全等价。

**建议 API**（即 fork 现有设计，可直接采纳）：
```rust
pub struct TurnRequest {
    // ...
    /// 追加在可缓存前缀之后的易变 system 片段。None = 无变化。
    pub ephemeral_system: Option<String>,
}
```

**验收**：`ephemeral_system: None` 的一轮，其请求体与该字段不存在时逐字节一致（不破坏现有缓存与快照测试）。

---

### R2 — 通用宿主反向 RPC（generic host reverse-RPC）

**动机**：某些 Tool 需要在执行中反向调用宿主拿数据（我们的场景是 Zotero，但这只是一个例子）。

**当前 fork 的问题**：fork 在 `snaca-tools-api` 里定义的 `ContextRequester` trait **把 Zotero 写死进了方法签名**：
```rust
// ❌ 上游不应包含领域方法
trait ContextRequester {
    async fn request_zotero_search(&self, query: &str, limit: Option<u32>) -> ...;
    async fn request_zotero_lookup(&self, key: &str) -> ...;
    async fn request_zotero_annotations(&self, item_key: &str) -> ...;
    async fn request_zotero_read(&self, key: &str) -> ...;
}
```
若照搬进上游，上游就被绑定到了 Zotero，违背原则 1。

**需求**：上游只提供一个**领域无关**的反向 RPC trait——不透明的方法名 + JSON 载荷。Zotero 的具体方法名与 schema 完全留给下游。

**建议 API**：
```rust
// snaca-tools-api
#[async_trait]
pub trait HostContext: Send + Sync + std::fmt::Debug {
    /// method 是宿主与下游 Tool 约定的字符串（如 "zotero.search"）。
    /// params/返回值对 snaca 都是不透明 JSON。
    async fn call(&self, method: &str, params: serde_json::Value)
        -> Result<serde_json::Value, HostContextError>;
}

#[derive(Debug, thiserror::Error)]
pub enum HostContextError {
    #[error("host rejected request: {0}")] HostRejected(String),
    #[error("timed out")] Timeout,
    #[error("host context unavailable: {0}")] Unavailable(String),
    #[error("invalid response payload: {0}")] InvalidPayload(String),
}
```
引擎侧提供一个**每轮工厂**的注入点（fork 已有等价物 `with_context_requester_factory`，泛化命名即可）：
```rust
// snaca-engine
pub type HostContextFactory =
    Arc<dyn Fn(/* turn_id */ String) -> Arc<dyn HostContext> + Send + Sync>;

impl Engine {
    pub fn with_host_context_factory(self, f: HostContextFactory) -> Self;
}
```

**验收**：不设置 factory 时，`ToolContext::host_context()`（见 R3）返回 `None`，所有现有工具行为不变。下游能在自己的 crate 里实现 `HostContext` 并注入，无需改 snaca。

---

### R3 — Tool 侧获取宿主句柄（tool-side accessor）

**动机**：R2 注入的 `HostContext` 需要能被 Tool 在执行时取到。

**需求**：`ToolContext` 提供一个泛型访问器返回 R2 的句柄；`with_*` 注入器供引擎装配。**不得**出现任何领域方法。

**建议 API**（对 fork 现有 `context_requester()` 泛化）：
```rust
impl ToolContext {
    pub fn with_host_context(self, h: Arc<dyn HostContext>) -> Self;
    pub fn host_context(&self) -> Option<&Arc<dyn HostContext>>;
}
```
> 备注：`ToolContext` 的 `Inner` 增加一个字段会迫使所有手写构造器更新。建议上游顺手把 `Inner` 改为 `#[derive(Clone)]` 并用 `(*self.inner).clone()` 收敛构造，避免每加一个字段就要改多处（fork 已被迫这么做，且能**降低**未来冲突）。

**验收**：下游 Zotero 工具形如 `ctx.host_context().ok_or(...)?.call("zotero.search", json!({...})).await`，全部落在下游 crate。

---

### R4 — 显式工作目录（explicit tool cwd）

**动机**：编辑器宿主希望文件类工具（Read/Write/Bash）的 cwd 指向用户**真实项目目录**，同时 SNACA 的元数据（memory/skills/db）仍落在 `data_root` 下。

**需求**：`WorkspaceLayout` 提供一个覆盖 `workspace_dir()` 的 builder；元数据路径不受影响。

**建议 API**（即 fork 现有设计，可直接采纳）：
```rust
impl WorkspaceLayout {
    /// 把工具 cwd 钉到一个绝对目录，覆盖派生的 workspace_dir。
    /// 元数据路径仍按 data_root 解析。
    pub fn with_explicit_workspace(self, dir: impl Into<PathBuf>)
        -> Result<Self, WorkspaceError>;
}
```

**验收**：不调用该 builder 时，`workspace_dir()` 行为与现在完全一致。

---

### R5 — thread 标题与 turn 分组（存储层）

**动机**：编辑器模式需要「可重命名的会话列表」和「按 turn 分组的消息」，上游 IM 模式两者都不存。这是本需求里唯一动到 schema 的部分。

上游有两种做法，请择一：

**方案 A（推荐）：作为一等字段收编**
这两项对任何宿主 UI 都通用（会话标题、turn 关联），不是 SciPen 特有，建议直接进上游主 schema：
- `threads` 增加 `title TEXT NOT NULL DEFAULT 'New conversation'`
- `messages` 增加 `turn_id TEXT`（可空）
- 对应 `NewThread { title }`、`ThreadRow { title }`、`NewMessage { turn_id }`、`MessageRow { turn_id }`
- 新增聚合查询 `ThreadSummaryRow { thread, last_active_at, turn_count }` 供会话列表
- 新增 `update_thread_title(...)`
- 旧库 in-place 迁移（幂等 `ALTER TABLE ADD COLUMN`），与上游既有的 `migrate_*` 风格一致

**方案 B：通用扩展元数据**
若上游不愿把这些纳入核心模型，则提供一张**下游可写的 sidecar 元数据表**（上游已有 `thread_compactions` 这种 1:1 sidecar 先例）：
```sql
-- 通用 kv，thread/message 维度各一张，值为不透明 JSON
CREATE TABLE thread_meta  (thread_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE message_meta (message_id TEXT PRIMARY KEY, data TEXT NOT NULL);
```
并暴露读写 API。下游把 `title`/`turn_id` 塞进 `data`。代价：热路径多一次 JOIN；`turn_id` 用 1:1 sidecar 建模不如可空列，故 R5 更倾向方案 A。

**验收**：全新库与旧库都能拿到带默认值的字段；不使用这些字段的现有部署行为不变。

---

### R6 — 注册自定义 Tool（extensible tool registry）

**动机**：下游要加自己的工具（Zotero 那组）而不编辑 `snaca-tools/src/lib.rs`。

**当前 fork 做法**：在 `default_*_registry_builder()` 里硬加 `add_zotero_tools(b)`——这要求改 snaca 源码。

**需求**：公开一个可组合的 registry builder，使下游能「取标准工具集 + 追加自己的 `Tool`」。`Tool` trait 及 `ToolRegistryBuilder` 视为公共 API。

**建议 API**：
```rust
// snaca-tools —— 标准集以 builder 形式暴露，可继续 .add()
pub fn default_registry_builder(/* 现有参数 */) -> ToolRegistryBuilder;

impl ToolRegistryBuilder {
    pub fn add<T: Tool + 'static>(self, tool: T) -> Self;
    pub fn build(self) -> ToolRegistry;
}
```
下游：`default_registry_builder(..).add(ZoteroSearchTool).add(ZoteroReadTool).build()`，全部在下游 crate。

> 若上游已有等价的公开 builder，则本需求只是「确认其为稳定公共 API 并允许追加任意 `Tool`」。

**验收**：下游不改 snaca，即可让自定义工具进入引擎所用的 registry。

---

### R7 — 注入自定义 LLM provider

**动机**：下游需要一个「OpenAI 兼容」provider（与 `deepseek` 平级），而不在 `snaca-llm` 内新增模块。

**当前 fork 做法**：在 `snaca-llm/src/` 里加 `openai` 模块并在 `lib.rs` 导出——要改 snaca 源码。

**需求**：
1. `LlmClient` trait、`ProviderCaps`、`MessageRequest/Response`、`RetryingLlmClient` 等视为稳定公共 API（据观察已 `pub use`，请正式承诺 semver）。
2. 引擎/会话装配接受注入 `Arc<dyn LlmClient>`，而非在内部按枚举选择 provider。

**建议**：下游在自己的 crate 里实现 `impl LlmClient for OpenAiClient`，构造引擎时注入。snaca 不需知道 openai 的存在。

**验收**：下游能仅凭公共 trait 实现并注入一个 provider，跑通一整轮对话，无需改 snaca。

---

## 4. 下游自行搬出 snaca 树的部分（无需上游改代码，只依赖 R1–R7 的公共 API）

以下 fork 内容会移出 snaca、迁入 SciPen Studio 自有仓库，作为依赖 snaca（submodule 内 path/crate 依赖）的下游 crate：

- **M1 — `snaca-editor` / `snaca-editor-protocol`**：编辑器 sidecar 及其 wire protocol（约 8,800 行）。它们目前作为 workspace 成员依赖 10+ 个 snaca 库 crate；搬出后改为依赖 submodule 内的 snaca crate。**前置条件**：这些被依赖的 snaca crate 的公共 API 足以支撑（当前基本满足，个别处依赖上文 R1–R4 的扩展点）。
- **M2 — Zotero 工具**（`snaca-tools/src/zotero.rs`）：改为下游 crate 内实现 `Tool` + `HostContext` 调用，靠 R2/R3/R6 接入。
- **M3 — OpenAI provider**（`snaca-llm/src/openai/`）：改为下游 crate 内实现 `LlmClient`，靠 R7 接入。

> 需上游确认：snaca 各 library crate 是否愿意/能够被 workspace **之外**的项目作为依赖引用（path 依赖指向 submodule 子目录，或发布到 registry）。这是 M1 的硬前置。

---

## 5. 版本与兼容

- R1–R7 均为**加法**，配合中性默认值，对现有 IM 模式部署零行为影响。
- 请在一个 minor 版本内落地并在 CHANGELOG 标注「下游可零改动集成」的 API 面。
- 下游会将 submodule 锁定到该 tag。

## 6. 完成的验收标准（下游视角）

达成后，SciPen Studio 应能：

1. 以 `git submodule add https://github.com/scipenai/snaca` 引入**未修改**的 snaca；
2. `git diff` 对比 submodule 与上游 tag **为空**；
3. 全部编辑器/Zotero/OpenAI 功能由下游 crate 通过 R1–R7 的公共 API 组合实现；
4. 升级 snaca = 更新 submodule 指针 + 处理 semver 级别的 API 变更，**不再有源码级 rebase 冲突**。

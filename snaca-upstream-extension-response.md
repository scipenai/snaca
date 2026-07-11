# SNACA 上游扩展需求 — 维护者回复

> 回复对象：SciPen Studio（下游宿主，编辑器模式集成方）
> 回复方：`scipenai/snaca` 维护者
> 对应文档：`snaca-upstream-extension-requirements.md`（R1–R7 / M1–M3）
> 版本基线：snaca **0.2.8**

---

## 0. 总体结论

我们**认同这个方向**：让下游以 submodule 引入未修改的 snaca，全部宿主逻辑通过公共扩展点组合。你们提的三条设计原则（上游不感知下游领域概念、加法优先+中性默认、稳定库边界）我们完全接受，并会作为后续演进的准绳。

分三类处置：

- **已交付**：R5 已在 **v0.2.8** 落地（下方第 2 节，含 API 与用法）。
- **成本极低、可快速落地**：R6、R7（上游架构已基本就绪，见第 3 节）；R1、R3、R4（干净的加法，见第 4 节）。
- **需先对齐设计再动手**：R2（与既有 `snaca-mcp` 的职责边界）、以及贯穿全局的 **semver 承诺范围**（见第 5 节）—— 这是本次唯一实质性的谈判点。

---

## 1. 逐项处置一览

| 编号 | 扩展点 | 处置 | 目标版本 | 备注 |
|---|---|---|---|---|
| **R5** | thread 标题 / turn 分组 | ✅ **已交付（sidecar 方案）** | 0.2.8 | 见第 2 节 |
| **R6** | 注册自定义 Tool | ✅ 采纳 | 0.3.0 | 通用部件已 `pub`，只需公开合成 builder |
| **R7** | 注入自定义 LLM provider | ✅ 采纳（已基本可用） | 0.3.0 | 引擎已接受 `Arc<dyn LlmClient>`，剩 semver 承诺 |
| **R1** | 每轮易变 system 上下文 | ✅ 采纳 | 0.3.0 | 契合现有 `SystemSegment` cacheable/volatile 机制 |
| **R3** | Tool 取宿主句柄 | ✅ 采纳（依赖 R2） | 0.3.0 | 照搬 `ToolContext` 现有不透明槽模式 |
| **R4** | 显式工作目录 | ✅ 采纳 | 0.3.0 | 真实缺口，`single_project()` 未覆盖该场景 |
| **R2** | 通用宿主反向 RPC | 🟡 采纳方向，先对齐实现方式 | 0.3.x | 需确认是否复用 MCP，见第 5.1 节 |
| **M1–M3** | 下游搬出 snaca 树 | ✅ 前置成立 | — | 见第 6 节 |

---

## 2. 已交付：R5（v0.2.8）

采纳你们建议里**倾向的方向的对立面**——不是把 `title`/`turn_id` 收进 core schema（方案 A），而是走 **sidecar 元数据表（方案 B）**。理由：让上游 schema 对下游领域概念**零感知**，严格符合原则 1。上游没有 `turn` 概念，若把 `turn_id` 塞进 `messages` 主表，等于让上游永久背一个自己从不写入的列；sidecar 把这份耦合彻底留在下游。

实现照搬了上游既有的 `thread_compactions` 1:1 sidecar 先例，风险可控。

### 新增存储（`snaca-state`）

```sql
CREATE TABLE IF NOT EXISTS thread_meta  (thread_id  TEXT PRIMARY KEY, data TEXT NOT NULL,
    FOREIGN KEY (thread_id)  REFERENCES threads(id)  ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS message_meta (message_id TEXT PRIMARY KEY, data TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE);
```

- `data` 对 snaca 是**不透明 JSON**；下游把 `title`/`turn_id` 等塞进去。
- 两表随 `schema.sql` 发布，`run_migrations` 通过 `CREATE TABLE IF NOT EXISTS` 对**新库与旧库都幂等建表**，无需迁移步骤；`ON DELETE CASCADE` 保证 thread/message 删除时元数据自动清理。

### 新增 `Database` API

```rust
// 不透明元数据读写（整块 JSON upsert，latest-wins）
async fn set_thread_meta(&self, thread: &ThreadId, data: &serde_json::Value) -> StateResult<()>;
async fn get_thread_meta(&self, thread: &ThreadId) -> StateResult<Option<serde_json::Value>>;
async fn set_message_meta(&self, message: &MessageId, data: &serde_json::Value) -> StateResult<()>;
async fn get_message_meta(&self, message: &MessageId) -> StateResult<Option<serde_json::Value>>;
// 批量读，避免渲染 turn 分组时 N+1
async fn get_message_meta_for_thread(&self, thread: &ThreadId)
    -> StateResult<Vec<(MessageId, serde_json::Value)>>;
// 会话列表聚合
async fn list_thread_summaries(&self, tenant: &TenantId, project: &ProjectId)
    -> StateResult<Vec<ThreadSummaryRow>>;
```

```rust
pub struct ThreadSummaryRow {
    pub thread: ThreadRow,
    pub last_active_at: Option<DateTime<Utc>>, // MAX(messages.created_at)
    pub message_count: u64,
    pub turn_count: u64,                         // COUNT(DISTINCT messages.session_id)
    pub meta: Option<serde_json::Value>,         // thread_meta.data 原始 JSON，上游不解析
}
```

**关于 `turn_count`**：上游没有 turn 概念，但每轮 `handle_turn` 会给该轮所有消息盖同一个 `session_id`，这是"轮"的天然持久化替身。因此聚合查询用 `COUNT(DISTINCT session_id)`，**不去解析 JSON 里的 `turn_id`**（否则就把 key 名写死进上游了）。若你们需要自己定义的 `turn_id` 分组，用 `get_message_meta_for_thread` 批量取回后在下游侧分组即可。

### 稳定入口（`snaca-sdk`）

`snaca-sdk` re-export 了 `Database, MessageRow, ThreadRow, ThreadSummaryRow, StateError, StateResult`，作为下游依赖的单一入口（`ThreadId`/`MessageId` 已由 `snaca_core` re-export 提供）。下游只依赖 `snaca-sdk` 即可调用上述全部 sidecar 方法。

### 端到端验证

仓库内 `examples/sdk/r5_sidecar_downstream.rs` 演示了完整下游流程（拥有自己的 `Database` → 跑真实对话轮 → 写标题 → 打 turn 标签 → 拉会话列表 + turn 分组），全程**不 import 任何 snaca 内部 crate**。已用真实 provider 跑通两轮对话，`list_thread_summaries` 正确回传 `title` / `turn_count=2` / `message_count` / `last_active_at`。

> 兼容性：R5 纯加法，两表不参与既有 `threads`/`messages` 查询，现有 IM 模式部署**逐字节零行为变化**。请把 submodule 锁到 tag **`v0.2.8`**。

---

## 3. 成本极低、可快速落地：R6、R7

这两项上游架构**已基本就绪**，本质上只差"确认为稳定公共 API"。

### R7 — 注入自定义 LLM provider（几乎已完成）

现状即你们想要的形态：
- `Engine` 内部持有 `llm: Arc<dyn LlmClient>`，`Engine::new` 的首参就是它；
- SDK 侧 `EngineRuntimeBuilder::llm_arc(Arc<dyn LlmClient>)` 可直接注入；
- provider 的 enum 判定只存在于 SDK helper `LlmOptions::build` 里，**引擎完全不参与**。

也就是说，下游在自己的 crate 里 `impl LlmClient for OpenAiClient`，构造引擎时 `.llm_arc(...)` 注入即可，snaca 无需知道 openai 的存在。**剩下的唯一动作是把 `LlmClient` / `ProviderCaps` / `MessageRequest`/`MessageResponse` / `RetryingLlmClient` 正式承诺为 semver 公共 API**（范围见第 5.2 节）。

### R6 — 可组合的 Tool registry

`Tool` trait、`ToolRegistry`、`ToolRegistryBuilder`（含 `.add()` / `.add_arc()` / `.builder()`）在 `snaca-tools-api` 里**已经全是 `pub`**。缺口仅在 `snaca-tools` 侧那些**合成好的** builder（`base_tool_registry_builder` 等）目前是 crate-private，导致下游无法"取标准工具集 + 追加自己的 Tool"。

我们会把这些合成 builder 公开为稳定 API，形如：

```rust
pub fn default_registry_builder(/* 现有参数 */) -> ToolRegistryBuilder;
// 下游：default_registry_builder(..).add(ZoteroSearchTool).add(ZoteroReadTool).build()
```

---

## 4. 干净的加法：R1、R3、R4

均为加法、中性默认，与现有机制契合，计划一并进 0.3.0。

- **R1（ephemeral_system）**：上游已实现 `SystemSegment { text, cacheable }` 的可缓存/易变分离，易变段每轮重算而不击穿缓存。我们会给每轮请求增加一个调用方可选的易变 system 片段，追加在可缓存前缀之后；`None` 时与现状逐字节一致（满足你们的验收）。这是对现有机制的自然延伸。

- **R3（ToolContext 取宿主句柄）**：`ToolContext` 的 `Inner` 已经持有一批 `Arc<dyn Any + Send + Sync>` 不透明槽（task_registry / question_gate / memory_provider…）。R2 的 `HostContext` 句柄会作为**同一模式**的又一个槽接入，提供 `with_host_context` / `host_context()` 访问器，不含任何领域方法。我们也会顺手把 `Inner` 收敛为可 clone 构造，降低你们后续 rebase 的冲突面。

- **R4（显式工作目录）**：这是**真实缺口**。上游现有 `single_project()` 模式虽能把 tool cwd 设为仓库根，但会强制把 metadata 塞进 `workspace_root/.snaca`，无法表达你们要的"cwd＝用户真实项目、metadata＝data_root"的解耦。我们会加一个 `WorkspaceLayout::with_explicit_workspace(dir)` builder，只覆盖 `workspace_dir()`，metadata 路径仍按 `data_root` 解析；不调用时行为与现状完全一致。

---

## 5. 需先对齐的两点

### 5.1 R2 —— 反向 RPC 的实现方式（新 trait vs 复用 MCP）

你们的设计判断是对的：上游**绝不能**把 `request_zotero_*` 这类领域方法写死进 trait。我们采纳"领域无关的不透明 method + JSON 载荷"这个形态。

唯一想先确认的是**实现载体**：snaca 已经有 `snaca-mcp` crate。"Tool 执行中反向调用宿主拿数据"这件事，能否把宿主建模成一个 Tool 侧可访问的 MCP server，从而复用既有机制，而不是新增一套 `HostContext` trait + 每轮工厂？

- 若能复用 MCP：维护面更小，不引入第二套扩展机制。
- 若 MCP 在"Tool→宿主"这个反向、低延迟、每轮上下文相关的场景下确实笨重：我们就采纳你们建议的独立 `HostContext` trait（形如 `async fn call(&self, method: &str, params: Value) -> Result<Value, HostContextError>`）+ `Engine::with_host_context_factory` 注入点。

**请给一段说明**：你们的 Zotero 反向调用在延迟/调用频率/是否需要 turn 级上下文上的实际约束，帮助我们二选一。R3 会跟随 R2 的最终形态落地。

### 5.2 贯穿全局的核心问题 —— semver 承诺范围

这是本次**最大的持续成本**，也是唯一需要你们和我们共同拍板的事。文档（R7-1 / M1 前置）实际要求把 `LlmClient`、`ProviderCaps`、`MessageRequest/Response`、`Tool`、`ToolRegistryBuilder`、`ToolContext`、`WorkspaceLayout`、`TurnRequest`、state 各 Row 结构体……这一大片都作为 semver 公共 API 承诺下来。让一个仍在 **0.x（pre-1.0）** 的项目冻结这么大的面，比新增 7 个扩展点本身更重。

我们的反提案：**收窄稳定面到一个显式的 facade**。

- 把 `snaca-sdk`（及其明确 re-export 的类型）声明为**稳定公共 API**，遵循 semver，在 CHANGELOG 标注"下游可零改动集成"的 API 面。
- 内部 library crate（`snaca-engine`/`snaca-state`/`snaca-tools` 等的**非** re-export 部分）**不纳入** semver 承诺，可以自由重构。
- 你们 M1 的编辑器 sidecar 若确实需要直接依赖某个内部 crate 的类型（而非经 sdk），请列一份**最小必需清单**，我们逐个评估：能上浮到 sdk facade 的就上浮，其余按"尽力保持、破坏时在 CHANGELOG 显著标注"处理。

这样上游保留 pre-1.0 的演进自由，你们也拿到一个明确、可锁定的稳定边界。**请确认是否接受"以 sdk facade 为 semver 边界"这一范围**——这是 R6/R7 正式落地前需要敲定的前提。

---

## 6. M1–M3 前置确认

- **library crate 可被 workspace 之外引用**：成立。各 snaca crate 已用 `path + version` 声明且带 `repository`，既能作为 submodule 内的 path 依赖被引用，也具备发布到 registry 的形态。M1 的硬前置满足。
- **M2（Zotero 工具）**：靠 R2/R3/R6 接入，实现留在下游 crate。
- **M3（OpenAI provider）**：靠 R7 接入，`impl LlmClient` 留在下游 crate。R7 现状已足以支撑（见第 3 节）。

唯一提醒：M1 编辑器 sidecar 依赖的内部 crate 类型，请配合第 5.2 节列出最小清单，以便我们规划稳定面。

---

## 7. 建议的落地顺序

1. **v0.2.8（已发布）**：R5 sidecar。你们可立即锁定该 tag，开始把存储层相关代码搬出 snaca 树。
2. **对齐环节（并行进行，不阻塞代码）**：
   - 确认 5.2 的 semver 范围（以 sdk facade 为边界）。
   - 给出 5.1 的 R2 实现方式所需的约束说明。
   - 提供 M1 依赖的内部类型最小清单。
3. **v0.3.0**：一次性落地 R1、R3、R4、R6、R7（均为加法，配中性默认，对现有 IM 模式零行为影响），并在 CHANGELOG 标注稳定 API 面。
4. **v0.3.x**：按 5.1 结论落地 R2（及跟随的 R3 最终形态）。

达成后即满足你们第 6 节的验收：submodule 与上游 tag `git diff` 为空，全部编辑器/Zotero/OpenAI 功能由下游 crate 通过公共 API 组合实现，升级 snaca = 更新指针 + 处理 semver 级变更，不再有源码级 rebase 冲突。

期待你们对 **5.1 / 5.2** 的回复，我们即可排 0.3.0。

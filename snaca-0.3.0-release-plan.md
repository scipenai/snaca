# SNACA 0.3.0 发版计划 —— 让 SciPen Studio 零改动 submodule 引入

> 基线：0.2.8（R5 已交付）
> 目标版本：**0.3.0**（minor，加法为主）
> 北极星：达成下游 `snaca-upstream-extension-requirements.md` §6 的验收——
> **submodule 与上游 tag `git diff` 为空**，全部 editor / Zotero / OpenAI 功能由下游 crate 只经 `snaca-sdk` 公共面组合实现。

---

## 0. 范围决定

第 3 轮已把 R2 的设计锁定为独立 `HostContext`（不再有待对齐的设计问题），因此**把 R2/R3 一并纳入 0.3.0**——否则 M2（Zotero 工具）搬不出 snaca 树，"直接 submodule 引入全部功能"不成立。0.3.0 = **R1 + R2 + R3 + R4 + R6 + R7 + M1 facade 上浮 + 稳定面固化**。

**冻结前仍待下游确认两点**（第 3 轮已抛出，见 §7）：mcp 收窄是否够用、`MemoryError` `#[non_exhaustive]` 是否可接受。二者有默认值，不阻塞开工，只影响 facade 最终符号集。

---

## 1. 工作流总览

| 编号 | 工作流 | 触及 crate | 类型 | 依赖 |
|---|---|---|---|---|
| **WS-1** | R7 provider 注入的 semver 承诺 | snaca-sdk | 承诺+验证（代码几近现成） | — |
| **WS-2** | R6 可组合 tool registry | snaca-tools, snaca-sdk | 加法（改可见性） | — |
| **WS-3** | R1 每轮 ephemeral_system | snaca-engine, snaca-llm | 加法 | — |
| **WS-4** | R4 显式工作目录 | snaca-workspace, snaca-sdk | 加法 | — |
| **WS-5** | R2 `HostContext` + R3 访问器 + 引擎工厂 | snaca-tools-api, snaca-engine, snaca-sdk | 加法 | — |
| **WS-6** | M1 facade 上浮（engine/state/workspace/skills/mcp/memory） | snaca-sdk | 加法（re-export + 收窄） | WS-2/4/5 |
| **WS-7** | 稳定面固化 + 边界门 + doc/CHANGELOG | scripts, snaca-sdk, snaca-state | 工程化 | 全部 |
| **WS-8** | 下游全功能验收 harness（"editor 模拟"示例） | examples/ | 验证 | 全部 |
| **WS-9** | 版本 bump 0.3.0 + tag | Cargo.toml/lock | 发版 | 全部 |

---

## 2. 各工作流详情

### WS-1 — R7 provider 注入（几乎现成，重点是承诺+守护）

现状即目标形态：`Engine` 持 `Arc<dyn LlmClient>`，`EngineRuntimeBuilder::llm_arc(...)` 可注入，provider enum 判定只在 SDK helper。**代码改动几乎为零**，本工作流的产出是：
- 把 `LlmClient`、`ProviderCaps`、`MessageRequest/MessageResponse`、`RetryingLlmClient`、`RetryConfig`、`StreamEvent`、`ContentDelta`、`LlmError/LlmResult`、`StopReason` 确认为 facade 稳定面（大多已 `pub use`，补齐缺项）。
- 加一个 facade 快照测试（见 WS-7）锁住这些符号。

**验收**：下游 `impl LlmClient for OpenAiClient` 后经 `AgentBuilder::llm_arc(...)` / `EngineRuntimeBuilder::llm_arc(...)` 注入，跑通整轮，snaca 不知 openai 存在。（M3 闭合）

### WS-2 — R6 可组合 tool registry

- `snaca-tools`：把合成好的 builder 由 crate-private 提升为 `pub`，对外形如
  `pub fn default_registry_builder(/* 现参 */) -> ToolRegistryBuilder;`（保留现有 `default_*_registry()` 便捷函数不变）。
- 确认 `SkillTool` `pub`（下游若要复用）。
- `snaca-sdk`：re-export `ToolRegistryBuilder`（`Tool`/`ToolRegistry`/`ToolContext` 已在 facade）+ 一个 `tools::default_builder()` 门面。

**验收**：下游 `default_registry_builder(..).add(ZoteroSearchTool).add(ZoteroReadTool).build()` 全落在下游 crate。（M2 的 registry 侧闭合）

### WS-3 — R1 每轮 ephemeral_system

- `snaca-engine`：`TurnRequest` 增 `pub ephemeral_system: Option<String>`（[engine.rs](crates/snaca-engine/src/engine.rs) 结构体 + `handle_turn_full` 解构处）。
- 在 `compose_system_segments`（engine.rs 现有 cacheable/volatile 装配处）末尾，当 `Some` 时追加一个 **volatile `SystemSegment`**，位于可缓存前缀之后。`None` 时不追加。
- `snaca-sdk`：`AgentInput` 增 `ephemeral_system(impl Into<String>)`，透传到 `TurnRequest`。

**验收**：`ephemeral_system: None` 的一轮，请求体与该字段不存在时**逐字节一致**（不破坏现有缓存/快照测试）；`Some` 时该片段每轮重算而不进可缓存前缀。加针对性测试。

### WS-4 — R4 显式工作目录

- `snaca-workspace`：`WorkspaceLayout` 增 `pub fn with_explicit_workspace(self, dir: impl Into<PathBuf>) -> Result<Self, WorkspaceError>`，只覆盖 `workspace_dir()` 的返回，metadata 路径（memory/skills/settings）仍按 `data_root`/`project_root` 解析。校验绝对路径（复用现有 `RootNotAbsolute`）。
- `snaca-sdk`：`AgentBuilder` 增 `explicit_workspace(dir)`；re-export `WorkspaceLayout` + `WorkspaceError`。

**验收**：不调用该 builder 时 `workspace_dir()` 行为与现状完全一致；调用后 tool cwd 指向真实项目目录、而 memory/db/skills 仍落 `data_root`。加测试断言两类路径解耦。

### WS-5 — R2 `HostContext` + R3 访问器 + 引擎工厂

- `snaca-tools-api`：定义领域无关的
  ```rust
  #[async_trait] pub trait HostContext: Send + Sync + std::fmt::Debug {
      async fn call(&self, method: &str, params: serde_json::Value)
          -> Result<serde_json::Value, HostContextError>;
  }
  #[derive(Debug, thiserror::Error)] #[non_exhaustive]
  pub enum HostContextError { HostRejected(String), Timeout, Unavailable(String), InvalidPayload(String) }
  ```
  `ToolContext` 增一个不透明槽（照搬 `Inner` 现有 `Arc<dyn Any>` 模式）+ `with_host_context(Arc<dyn HostContext>)` / `host_context() -> Option<&Arc<dyn HostContext>>`。顺手把 `Inner` 收敛为可 clone 构造，降低下游 rebase 冲突面。
- `snaca-engine`：`type HostContextFactory = Arc<dyn Fn(String /*turn_id*/) -> Arc<dyn HostContext> + Send + Sync>;` + `Engine::with_host_context_factory(f)`；每轮用 `turn_id` 造句柄注入 `ToolContext`。
- `snaca-sdk`：re-export `HostContext`、`HostContextError`、`HostContextFactory`；`AgentBuilder`/`EngineRuntimeBuilder` 增注入点。

**验收**：不设 factory 时 `host_context()` 返回 `None`、现有工具行为不变；下游在自己 crate 实现 `HostContext`（传输＝其 editor 的 `OutboundWriter`+correlator），Tool 内 `ctx.host_context().ok_or(..)?.call("zotero.search", json!({..})).await`，全落下游。（M2 的反向 RPC 侧闭合）

### WS-6 — M1 facade 上浮

单一原则：**下游 `snaca-editor` 最终只依赖 `snaca-sdk`（+ 其 `snaca-editor-protocol`）**。在 `snaca-sdk` re-export：

- **engine**：`Engine`、`EngineConfig`、`NoopApprovalGate`、`TurnEventListener`
- **state**：`NewThread`、`NewMessage`（补 `Database` 写入侧；R5 的读侧已在）
- **workspace**：`WorkspaceLayout` + `WorkspaceError`（随 WS-4）
- **skills**：新增 `snaca_sdk::skills` 模块 → `Skill`、`SkillRegistry`、`SkillRegistryBuilder`、`SkillScope`、`SkillProvider`（+ `SkillError/SkillResult/DynSkillProvider`）。新增 sdk→skills 依赖边（边界脚本允许）。
- **mcp（收窄）**：新增 `snaca_sdk::mcp` 模块 → **仅** `McpManager`、`McpServerConfig`、`McpTransport`。**不导出** `McpClient`/`McpTool`（其签名泄漏 `rmcp` 外部类型，会绑死 semver）。新增 sdk→mcp 依赖边。
- **memory（收窄）**：`snaca_sdk::memory` 补 store 侧 `MemoryStore`、`MemoryScope`、`MemoryError`；给 `MemoryError` 加 `#[non_exhaustive]`。

> 二处收窄若下游不接受（§7），则该组改走"下游直接依赖内部 crate + CHANGELOG 破坏标注"的 5.2 兜底，不影响其余上浮。

### WS-7 — 稳定面固化 + 边界门

- **facade 快照测试**：在 `snaca-sdk` 加一个编译期测试，`use snaca_sdk::{...};` 显式列出全部承诺符号；任何误删/改名会让该测试编译失败（等价一份可执行的 API 清单）。
- **facade-only 门**：扩展 `scripts/check-sdk-boundaries.sh` —— 新增校验 `examples/`（下游 harness，见 WS-8）**不出现** `snaca_engine::`/`snaca_state::`/`snaca_workspace::`/`snaca_skills::`/`snaca_mcp::`/`snaca_memory::` 等直接 import（只允许 `snaca_sdk::`）。这把"零改动 submodule＝只经 facade"变成 CI 可执行的不变量。
- **CHANGELOG**：新增 `## 0.3.0` 段，标注"下游可零改动集成"的稳定 API 面清单 + semver 承诺声明（facade 为边界）。
- **turn_count doc**：给 `ThreadSummaryRow.turn_count` 补一句"宿主每回合多次 handle_turn 时本计数会超过宿主 turn 数"。
- `make check-boundaries` 保持在 CI（已在 Makefile）。

### WS-8 — 下游全功能验收 harness（核心证据）

把 `examples/sdk/r5_sidecar_downstream.rs` 扩成一个 **`editor_like_downstream.rs`**，**只 import `snaca_sdk::*`**，一次性行使全部扩展点：

1. **R7**：自定义 `impl LlmClient`（OpenAI 兼容 stub 或真实 provider）经 `llm_arc` 注入。
2. **R4**：`explicit_workspace(<真实项目目录>)`，metadata 落 `data_root`。
3. **R6**：`default_registry_builder(..).add(<自定义 Tool>)`。
4. **R2/R3**：自定义 `impl HostContext`（内存 stub 模拟 editor 回调，含一个 `zotero.search` method）；自定义 Tool 内 `ctx.host_context()?.call("zotero.search", ..)`；经 `with_host_context_factory` 注入。
5. **R1**：每轮 `AgentInput::ephemeral_system("当前打开文件/选区…")`。
6. **R5**：`set_thread_meta`(title) + `set_message_meta`(turn_id) + `list_thread_summaries` + `get_message_meta_for_thread`。
7. **M1**：用 `snaca_sdk::skills` 组一个 `SkillRegistry`、用 `snaca_sdk::mcp::McpManager` 装一个 MCP server、用 `snaca_sdk::memory::MemoryStore` 直接读写。

该示例**编译并运行成功 + 通过 WS-7 的 facade-only 门 = 零改动 submodule 可行的机器可验证证明**。这就是 0.3.0 的最终验收。

### WS-9 — 发版

`[workspace.package] version` 0.2.8 → **0.3.0**；`cargo update --workspace` 刷新 lock；`chore(release): bump version to 0.3.0`；tag `v0.3.0`。

---

## 3. 验收标准 ↔ 下游 §6 映射

| 下游验收（§6） | 0.3.0 中的保证 |
|---|---|
| submodule 引入未修改 snaca | 全部扩展点为加法，无需改源 |
| submodule 与上游 tag diff 为空 | WS-8 harness 全在 `examples/`（上游侧）；下游真实代码在其自有仓库，只依赖 `snaca-sdk` |
| 全部 editor/Zotero/OpenAI 由公共面组合 | R1(editor 上下文)/R4(cwd)/R2+R3+R6(Zotero)/R7(OpenAI)/R5(会话)+M1(skills/mcp/memory) 全经 facade |
| 升级=更新指针+处理 semver 变更 | WS-7 把 facade 固化为 semver 边界 + CI 门 |

---

## 4. PR 拆分（建议顺序，彼此低耦合）

1. **PR-1 WS-3 R1 ephemeral_system**（引擎内，独立）
2. **PR-2 WS-4 R4 explicit workspace**（workspace + sdk）
3. **PR-3 WS-2 R6 registry builder**（tools 可见性 + sdk）
4. **PR-4 WS-5 R2/R3 HostContext**（tools-api + engine + sdk，最大一块）
5. **PR-5 WS-1+WS-6 facade 上浮 + provider 承诺**（sdk re-export，含 skills/mcp/memory 收窄）
6. **PR-6 WS-7 稳定面固化**（facade 快照测试 + boundary 脚本扩展 + CHANGELOG + doc）
7. **PR-7 WS-8 下游全功能 harness**（依赖前六者，最终验收）
8. **PR-8 WS-9 bump 0.3.0 + tag**

每个 PR 独立过 `cargo test --workspace` + `cargo clippy` + `cargo fmt --check` + `make check-boundaries`。（注：本轮起把 `fmt --check` 纳入本地验证——上次 CI 就栽在这。）

---

## 5. 兼容性与风险

- **纯加法 + 中性默认**：R1/R4/R5 的新字段/新 builder 不调用即与现状逐字节一致；facade 上浮只增不减；现有 IM 模式部署零行为影响。
- **`Engine::new` 签名进 semver**：承诺 `Engine` 意味着冻结其构造/方法签名——0.3.x 内避免破坏性改这些；若需改，minor 版本 + CHANGELOG。
- **新依赖边（sdk→skills/mcp）**：`check-sdk-boundaries.sh` 允许（只禁 sdk→server/channel-host）；会略增 sdk 编译面，可接受。
- **rmcp 泄漏**：靠 mcp 收窄挡在 facade 外——务必在 WS-6 落实"不 re-export `McpClient`/`McpTool`"。
- **`#[non_exhaustive]` 传导**：`MemoryError`、`HostContextError` 均加，下游需通配臂（已在第 3 轮说明）。

---

## 6. 待下游确认（冻结 facade 符号集的唯一前置）

1. **mcp 收窄**：`McpManager` + `McpServerConfig` + `McpTransport` 是否满足"自管 MCP servers"？否则该组走 5.2 兜底。
2. **`MemoryError` `#[non_exhaustive]`**：可接受通配臂，还是该组走 5.2 兜底？

默认取"收窄上浮"。两点回后，WS-6 的符号集即完全冻结，其余工作流不受影响，可立即开工。

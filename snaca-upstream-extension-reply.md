# SNACA 上游扩展需求 — 下游回复（第 2 轮）

> 回复对象：`scipenai/snaca` 维护者
> 回复方：SciPen Studio（下游宿主，编辑器模式集成方）
> 对应文档：`snaca-upstream-extension-response.md`（维护者回复，基线 0.2.8）
> 目的：回应你们提出的 **5.1（R2 实现方式）**、**5.2（semver 范围）**、**M1（最小内部类型清单）** 三项

---

## 0. 总体

感谢 v0.2.8 的 R5 交付与 `examples/sdk/r5_sidecar_downstream.rs`。我们已核对：示例全程只用 `snaca_sdk` 公共面 + sidecar API，schema 注释严格贯彻「上游对形状零感知」，`thread_meta`/`message_meta` 用 `CREATE TABLE IF NOT EXISTS` 对新旧库幂等建表、`ON DELETE CASCADE` 兜住生命周期——**完全符合我们的原则 1**。我们接受把 submodule 锁到 tag `v0.2.8`，即刻开始把存储层相关代码搬出 snaca 树。

也认可你们的落地顺序：**0.3.0 一次性落 R1/R3/R4/R6/R7，0.3.x 落 R2**。

本回复给出三件你们等待的事：

1. **5.1** — 我方 Zotero 反向调用的实测约束，以及基于此的**推荐：采纳独立 `HostContext` trait，不复用 MCP**。
2. **5.2** — **接受以 `snaca-sdk` facade 作为 semver 边界**，附我方唯一条件（把 M1 清单上浮进 facade）。
3. **M1** — 精确的最小内部类型清单（已按「建议归属」分组，并标注哪些已被 R4/R6 计划顺带覆盖）。

外加一个 **R5 使用澄清**（`turn_count` 与我方 composer 双-engine 的语义，见第 4 节）。

---

## 1. 回应 5.1 — R2 反向 RPC：建议采纳独立 `HostContext`，不复用 MCP

你们问的核心是「能否把宿主建模成 Tool 侧可访问的 MCP server 来复用既有机制」。下面先给**实测约束**（来自我方 fork 的现有实现），再给结论。

### 我方反向调用的实际形态

- **传输：复用编辑器已经打开的 JSON-RPC stdio 双工**。具体是 `OutboundWriter`（写到宿主）+ `ContextCorrelator`（按 id 关联回包）。**不是**另起的连接或进程。
- **turn 绑定**：每次调用都挂当前 `turn_id`，宿主据此记录 per-turn telemetry、并支持 turn 级 abort。
- **延迟 / parking**：per-call 5s 超时；**交互式**——宿主收到后可能弹 UI（下面会展开）。
- **频率**：每轮 0..N 次，模型驱动，**低频但延迟敏感**。
- **关键：同一条 `context.request` 通道承载的远不止 Zotero**。当前 payload 变体包括 `FileContent`、`FlushUnsaved`、`AskUserQuestion`、以及 4 个 `Zotero*`。也就是说这是一个通用的「引擎/Tool → 宿主，要一份东西、按 turn 关联拿回」的机制，Zotero 只是其中一类 method。

### 为什么 MCP 在这个场景不合适（逐条）

1. **方向反了**。MCP 的模型是 *snaca 作为 client 去连外部 tool server*。这里 counterparty 是**已经在同一条 stdio 通道另一端**的宿主。让宿主再起一个 MCP server endpoint，是为一次「回调」重复搭一套传输。
2. **没有 turn 概念**。我们需要每次调用绑定 `turn_id` 做 telemetry 与 abort；MCP 的 tool 调用不带 turn 语义。
3. **承载的是宿主 UI 交互回调，不是「数据源工具」**。`AskUserQuestion`（向用户提问）、`FlushUnsaved`（请宿主把编辑器未保存缓冲 flush 到磁盘）本质是**宿主动作**，不契合 MCP「tool = 可发现的数据源/能力」的模型。
4. **成本对比**。复用已开的 duplex ≈ 零额外传输；MCP 要额外的连接 + init 握手 + tool 发现，对一个 5s 交互回调是净负担。

### 结论与对上游的最小要求

**推荐采纳你们在 5.1 里给的备选形态——独立的领域无关 `HostContext`**：

```rust
#[async_trait]
pub trait HostContext: Send + Sync + std::fmt::Debug {
    async fn call(&self, method: &str, params: serde_json::Value)
        -> Result<serde_json::Value, HostContextError>;
}
```

- 上游**只需**：这个 trait 定义 + 每轮注入点 `Engine::with_host_context_factory(...)` + `ToolContext` 访问器（正是你们 R3 计划里「又一个不透明 `Arc<dyn Any>` 槽」的同一模式）。
- 上游**不需要**提供任何传输——传输是我方 editor 的事（`OutboundWriter` + correlator 留在下游）。
- 维护面因此**比复用 MCP 更小**，不是更大：没有第二套 server/传输，只是 `ToolContext` 多一个已有模式的槽。

> 补充：`method` 命名空间（如 `zotero.search`、`editor.file_content`）与 payload schema 完全由我方 editor 与我方 Tool 约定，snaca 全程当不透明 JSON。这样 R2 落地后，上游对「Zotero」「editor」零感知，严格符合原则 1。

---

## 2. 回应 5.2 — 接受以 sdk facade 作为 semver 边界

**我们接受你们的反提案**：以 `snaca-sdk`（及其显式 re-export 的类型）为唯一 semver 稳定面；内部 library crate 的**非** re-export 部分不纳入 semver，可自由重构。对一个 pre-1.0 项目，这个边界合理，我们也因此拿到一个明确、可锁定的稳定面——比要求你们冻结一大片内部 API 更可持续。

**我方唯一条件**：把第 3 节 M1 清单里的最小集合**上浮进 sdk facade**。原则按你们说的办——能上浮的上浮；个别实在不宜上浮的类型，我们接受「editor sidecar 直接依赖该内部 crate，但你们在破坏时于 CHANGELOG 显著标注」的兜底。

---

## 3. M1 — 最小内部类型清单（你们要的）

方法：枚举 `snaca-editor` 对内部 snaca crate 的全部 import，逐一比对 **v0.2.8 `snaca-sdk` 的公开 re-export 面**。下表只列**当前不在 sdk facade 内**的符号（已在 facade 的 `snaca_core::*`、agent_api 的 approval/question/memory provider 类型、`Database/ThreadRow/MessageRow/ThreadSummaryRow`、`StreamEvent/ContentDelta`、`ToolRegistry/Tool/ToolContext` 等**均已覆盖，不再列出**）。

| 符号 | 来源 crate | 用途 | 建议处置 |
|---|---|---|---|
| `Engine` | snaca-engine | 我方直接构造/持有引擎（composer 双-engine） | **上浮 sdk**（当前仅作 `Agent::engine()` 返回类型出现，未 `pub use`） |
| `EngineConfig` | snaca-engine | 直接配置引擎 | **上浮 sdk** |
| `NoopApprovalGate` | snaca-engine | 默认 approval gate | **上浮 sdk** |
| `TurnEventListener` | snaca-engine | 订阅流式 turn 事件 | **上浮 sdk** |
| `NewThread` / `NewMessage` | snaca-state | 我方直接写会话/消息历史 | **上浮 sdk**（已有 `Database` 但缺这两个写入结构体） |
| `WorkspaceLayout` | snaca-workspace | 装配工作区布局 | **上浮 sdk**（R4 本就要动它，顺带 re-export） |
| `ToolRegistryBuilder` | snaca-tools-api | 组合工具集 | **R6 已覆盖**（`default_registry_builder` 返回它，请一并 re-export 该类型） |
| `base_tool_registry` | snaca-tools | 取标准工具集起点 | **R6 已覆盖**（合成 builder 公开） |
| `SkillTool` | snaca-tools | 具体 skill 工具 | 请随 R6 一并 `pub` / re-export |
| `Skill`, `SkillRegistry`, `SkillRegistryBuilder`, `SkillScope`, `SkillProvider` | snaca-skills | 我方装配 skill 注册表 | **上浮 sdk**（建议新增 `snaca_sdk::skills` 模块） |
| `MemoryStore`, `MemoryScope`, `MemoryError` | snaca-memory | 我方直接读写 memory store | **上浮 sdk**（sdk 现有 `memory` 模块只暴露 provider 侧，缺 store 侧） |
| `McpManager`, `McpServerConfig`, `McpTransport` | snaca-mcp | 我方自管 MCP servers | **上浮 sdk**（建议新增 `snaca_sdk::mcp` 模块）——**清单里最大的一块** |

说明：

- `TurnRequest` 已被 sdk re-export 为 `EngineTurnRequest`，`RuntimeToolFactory`、approval/question gate 类型也已覆盖，故未列。
- 上述里 **`WorkspaceLayout`、`ToolRegistryBuilder`、`base_tool_registry`** 会被你们 **R4/R6 计划顺带闭合**，真正「净新增 facade 面」集中在 **engine（4 个）+ state（2 个）+ skills（5 个）+ memory（3 个）+ mcp（3 个）**。
- 一旦这些进 facade，**`snaca-editor` 可做到只依赖 `snaca-sdk`（+ 我方 `snaca-editor-protocol`）**，M1 的硬前置闭合，无需直接依赖任何内部 crate。

如果 skills/mcp/memory 这几组你们不愿全部上浮，请指出哪些留在内部，我们对这几个走 5.2 的兜底（直接依赖 + CHANGELOG 标注）。

---

## 4. R5 使用澄清 —— `turn_count` 与我方 composer 的双-engine

你们的 `list_thread_summaries` 用 `COUNT(DISTINCT session_id)` 作 `turn_count`，并说明「每轮 `handle_turn` 给该轮所有消息盖同一 `session_id`」。

**需要对齐的一点**：我方编辑器的 composer 会把一个用户回合拆成**两次 `handle_turn`**——一个 plan turn（独立 plan engine）+ 一个 exec turn。若这两次各自获得不同的 `session_id`，那么一个用户视角的「turn」在 `turn_count` 里会被计为 2。

这对我们**不是阻塞**：我们本来就用 `message_meta` 里自己写的 `turn_id` 作**权威分组**（正如你们建议、也如 example 里 `get_message_meta_for_thread` 的用法），`turn_count` 只当便利辅助。**只想确认双方理解一致**：`turn_count` 反映的是 snaca 的 `session_id` 计数（可能 ≠ 宿主定义的 turn），下游若要精确 turn 数应以 `message_meta.turn_id` 为准。若你们认同，这一点无需上游改动。

---

## 5. 落地顺序（确认）

1. **v0.2.8（已发布）**：锁定该 tag，开始搬存储层。✅ 认可。
2. **0.3.0**：R1、R3、R4、R6、R7 + **M1 facade 上浮**（第 3 节），CHANGELOG 标注稳定 API 面。
3. **0.3.x**：R2 —— 采纳独立 `HostContext`（第 1 节），R3 随其最终形态落地。

回到你们的两个待办：**5.1 我们给了明确推荐（独立 `HostContext`，附实测约束）**；**5.2 我们接受 sdk facade 边界，条件是 M1 清单上浮**。据此即可排 0.3.0。

期待你们对 **M1 清单上浮范围** 的确认（尤其 skills / memory / mcp 三组是否全部进 facade），以及对 **5.1 采用独立 `HostContext`** 的最终认可。

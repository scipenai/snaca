//! End-to-end test for the IM-attachment-import flow.
//!
//! Boots a real `Runtime` with a single mock plugin that:
//!   - auto-injects one `event.message_received` carrying an `Attachment`
//!   - serves that attachment's bytes when the host calls `file.download`
//!
//! The dispatcher should pull the attachment, run it through
//! `engine.import_attachment`, and write a memory entry under the
//! project's reference scope. We assert by reading the on-disk memory
//! tree once the round trip completes.

use async_trait::async_trait;
use snaca_core::{Message, MessageId, ProjectId, Role, TenantId, Usage};
use snaca_llm::{LlmClient, LlmResult, MessageRequest, MessageResponse, ProviderCaps, StopReason};
use snaca_memory::{MemoryScope, MemoryStore};
use snaca_server::{Config, Runtime};
use snaca_workspace::WorkspaceLayout;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn snaca_cli_binary() -> PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        let cargo = escargot::CargoBuild::new()
            .bin("snaca-cli")
            .package("snaca-cli")
            .current_target()
            .run()
            .expect("build snaca-cli");
        cargo.path().to_path_buf()
    })
    .clone()
}

struct ConstantLlm {
    text: String,
    calls: AtomicUsize,
    last_user_text: std::sync::Mutex<Option<String>>,
    user_texts: std::sync::Mutex<Vec<String>>,
}

impl ConstantLlm {
    fn new(text: &str) -> Self {
        Self {
            text: text.into(),
            calls: AtomicUsize::new(0),
            last_user_text: std::sync::Mutex::new(None),
            user_texts: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl LlmClient for ConstantLlm {
    fn provider_name(&self) -> &'static str {
        "constant-mock"
    }
    fn model(&self) -> &str {
        "constant"
    }
    fn capabilities(&self) -> ProviderCaps {
        ProviderCaps {
            tool_use: true,
            ..Default::default()
        }
    }
    async fn create_message(&self, req: MessageRequest) -> LlmResult<MessageResponse> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let last_user_text = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|b| match b {
                        snaca_core::ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            });
        if let Some(text) = last_user_text.clone() {
            self.user_texts.lock().unwrap().push(text);
        }
        *self.last_user_text.lock().unwrap() = last_user_text;
        Ok(MessageResponse {
            id: "constant".into(),
            message: Message {
                id: MessageId::new(),
                role: Role::Assistant,
                content: vec![snaca_core::ContentBlock::text(self.text.clone())],
                created_at: chrono::Utc::now(),
            },
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                ..Default::default()
            },
            stop_reason: StopReason::EndTurn,
        })
    }
}

#[tokio::test]
async fn attachment_lands_as_memory_entry_before_turn() {
    let _ = tracing_subscriber::fmt::try_init();
    let tmp = tempfile::tempdir().unwrap();
    let data_root = tmp.path().join("data");
    let cfg_path = tmp.path().join("snaca.toml");
    let cli_bin = snaca_cli_binary();

    // Inject one user message + one attachment. The dispatcher should
    // import the attachment first, then run the turn.
    let cfg = format!(
        r#"
[server]
http_listen = "127.0.0.1:0"
data_root = {data_root:?}

[tenant]
id = "default"

[llm]
api_key = "ignored"
model = "constant"

[engine]
memory_extractor = false

[[plugins]]
name = "mock"
command = {cli_bin:?}
args = [
    "mock-plugin",
    "--auto-inject",
    "look at the spec",
    "--inject-attachment",
    "att-1:spec.md:project conventions: kebab-case file names",
]
"#,
        data_root = data_root.to_string_lossy(),
        cli_bin = cli_bin.to_string_lossy(),
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let config = Config::load(&cfg_path).expect("config loads");
    let llm = Arc::new(ConstantLlm::new("noted"));
    let runtime = Runtime::build_with_llm(config, llm.clone())
        .await
        .expect("runtime starts");

    // The mock plugin's `inject_tenant_id`/`inject_chat_id` defaults
    // give us deterministic routing. Project is the chat-id-derived
    // auto slug.
    let tenant = TenantId::new("mock-tenant");
    let project = ProjectId::auto_from_chat("mock-chat");

    // Wait up to 10s for the attachment to land in the memory tree.
    // The dispatcher imports synchronously *before* the LLM turn, so
    // an entry under `reference/` is the success signal.
    let layout = WorkspaceLayout::new(std::fs::canonicalize(&data_root).unwrap()).unwrap();
    let store = MemoryStore::new(layout.memory_dir(&tenant, &project));

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut names = Vec::new();
    while Instant::now() < deadline {
        names = store.list(MemoryScope::Reference).await.unwrap_or_default();
        if !names.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        !names.is_empty(),
        "expected an attachment-derived memory entry; got: {names:?}"
    );
    let landed = names.iter().any(|n| n.contains("spec"));
    assert!(
        landed,
        "expected `spec`-derived entry; got names: {names:?}"
    );

    // Read the entry and confirm the inlined content actually made
    // it through file.download → import_one.
    let stem = names.iter().find(|n| n.contains("spec")).unwrap().clone();
    let entry = store.read(MemoryScope::Reference, &stem).await.unwrap();
    assert!(
        entry.content.contains("kebab-case"),
        "import did not preserve content; got: {:?}",
        entry.content
    );

    // The LLM was eventually invoked too — attachment import must
    // not block the turn from running.
    wait_for_llm_calls(&llm, 1).await;

    runtime.shutdown().await;
}

#[tokio::test]
async fn file_then_text_is_assembled_before_attachment_import() {
    let _ = tracing_subscriber::fmt::try_init();
    let tmp = tempfile::tempdir().unwrap();
    let data_root = tmp.path().join("data");
    let cfg_path = tmp.path().join("snaca.toml");
    let cli_bin = snaca_cli_binary();

    let cfg = format!(
        r#"
[server]
http_listen = "127.0.0.1:0"
data_root = {data_root:?}

[tenant]
id = "default"

[llm]
api_key = "ignored"
model = "constant"

[engine]
memory_extractor = false

[im_input]
text_debounce_ms = 50
attachment_wait_secs = 5
referential_text_wait_secs = 5
pending_expire_secs = 5

[[plugins]]
name = "mock"
command = {cli_bin:?}
args = [
    "mock-plugin",
    "--auto-inject",
    "[uploaded file: spec.md]",
    "--inject-attachment",
    "att-1:spec.md:project conventions: kebab-case file names",
    "--inject-extra",
    "mock-chat:请总结重点",
]
"#,
        data_root = data_root.to_string_lossy(),
        cli_bin = cli_bin.to_string_lossy(),
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let config = Config::load(&cfg_path).expect("config loads");
    let llm = Arc::new(ConstantLlm::new("noted"));
    let runtime = Runtime::build_with_llm(config, llm.clone())
        .await
        .expect("runtime starts");

    let tenant = TenantId::new("mock-tenant");
    let project = ProjectId::auto_from_chat("mock-chat");
    wait_for_spec_memory(&data_root, &tenant, &project).await;
    wait_for_llm_calls(&llm, 1).await;

    let user_text = llm
        .last_user_text
        .lock()
        .unwrap()
        .clone()
        .expect("LLM saw a user message");
    assert_eq!(user_text, "请总结重点");
    runtime.shutdown().await;
}

#[tokio::test]
async fn text_then_file_is_assembled_before_attachment_import() {
    let _ = tracing_subscriber::fmt::try_init();
    let tmp = tempfile::tempdir().unwrap();
    let data_root = tmp.path().join("data");
    let cfg_path = tmp.path().join("snaca.toml");
    let cli_bin = snaca_cli_binary();

    let cfg = format!(
        r#"
[server]
http_listen = "127.0.0.1:0"
data_root = {data_root:?}

[tenant]
id = "default"

[llm]
api_key = "ignored"
model = "constant"

[engine]
memory_extractor = false

[im_input]
text_debounce_ms = 50
attachment_wait_secs = 5
referential_text_wait_secs = 5
pending_expire_secs = 5

[[plugins]]
name = "mock"
command = {cli_bin:?}
args = [
    "mock-plugin",
    "--auto-inject",
    "帮我总结这个文件",
    "--inject-attachment",
    "att-1:spec.md:project conventions: kebab-case file names",
    "--inject-extra",
    "mock-chat:attachment:[uploaded file: spec.md]",
]
"#,
        data_root = data_root.to_string_lossy(),
        cli_bin = cli_bin.to_string_lossy(),
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let config = Config::load(&cfg_path).expect("config loads");
    let llm = Arc::new(ConstantLlm::new("noted"));
    let runtime = Runtime::build_with_llm(config, llm.clone())
        .await
        .expect("runtime starts");

    let tenant = TenantId::new("mock-tenant");
    let project = ProjectId::auto_from_chat("mock-chat");
    wait_for_spec_memory(&data_root, &tenant, &project).await;
    wait_for_llm_calls(&llm, 1).await;

    let user_text = llm
        .last_user_text
        .lock()
        .unwrap()
        .clone()
        .expect("LLM saw a user message");
    assert_eq!(user_text, "帮我总结这个文件");
    runtime.shutdown().await;
}

#[tokio::test]
async fn file_only_waits_for_instruction_without_autorun() {
    let _ = tracing_subscriber::fmt::try_init();
    let tmp = tempfile::tempdir().unwrap();
    let data_root = tmp.path().join("data");
    let cfg_path = tmp.path().join("snaca.toml");
    let cli_bin = snaca_cli_binary();

    let cfg = format!(
        r#"
[server]
http_listen = "127.0.0.1:0"
data_root = {data_root:?}

[tenant]
id = "default"

[llm]
api_key = "ignored"
model = "constant"

[engine]
memory_extractor = false

[im_input]
text_debounce_ms = 20
attachment_wait_secs = 1
referential_text_wait_secs = 1
pending_expire_secs = 5
file_only_autorun = false

[[plugins]]
name = "mock"
command = {cli_bin:?}
args = [
    "mock-plugin",
    "--auto-inject",
    "[uploaded file: spec.md]",
    "--inject-attachment",
    "att-1:spec.md:project conventions: kebab-case file names",
]
"#,
        data_root = data_root.to_string_lossy(),
        cli_bin = cli_bin.to_string_lossy(),
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let config = Config::load(&cfg_path).expect("config loads");
    let llm = Arc::new(ConstantLlm::new("noted"));
    let runtime = Runtime::build_with_llm(config, llm.clone())
        .await
        .expect("runtime starts");

    tokio::time::sleep(Duration::from_millis(1800)).await;
    assert_eq!(
        llm.calls.load(Ordering::Relaxed),
        0,
        "file-only pending input should not invoke the LLM without instructions"
    );

    let tenant = TenantId::new("mock-tenant");
    let project = ProjectId::auto_from_chat("mock-chat");
    let layout = WorkspaceLayout::new(std::fs::canonicalize(&data_root).unwrap()).unwrap();
    let store = MemoryStore::new(layout.memory_dir(&tenant, &project));
    let names = store.list(MemoryScope::Reference).await.unwrap_or_default();
    assert!(
        names.is_empty(),
        "file-only pending input should not import before submission; got {names:?}"
    );

    runtime.shutdown().await;
}

#[tokio::test]
async fn file_only_start_command_submits_pending_file() {
    let _ = tracing_subscriber::fmt::try_init();
    let tmp = tempfile::tempdir().unwrap();
    let data_root = tmp.path().join("data");
    let cfg_path = tmp.path().join("snaca.toml");
    let cli_bin = snaca_cli_binary();

    let cfg = format!(
        r#"
[server]
http_listen = "127.0.0.1:0"
data_root = {data_root:?}

[tenant]
id = "default"

[llm]
api_key = "ignored"
model = "constant"

[engine]
memory_extractor = false

[im_input]
text_debounce_ms = 20
attachment_wait_secs = 5
referential_text_wait_secs = 5
pending_expire_secs = 5
file_only_autorun = false

[[plugins]]
name = "mock"
command = {cli_bin:?}
args = [
    "mock-plugin",
    "--auto-inject",
    "[uploaded file: spec.md]",
    "--inject-attachment",
    "att-1:spec.md:project conventions: kebab-case file names",
    "--inject-extra",
    "mock-chat:开始处理",
]
"#,
        data_root = data_root.to_string_lossy(),
        cli_bin = cli_bin.to_string_lossy(),
    );
    std::fs::write(&cfg_path, cfg).unwrap();

    let config = Config::load(&cfg_path).expect("config loads");
    let llm = Arc::new(ConstantLlm::new("noted"));
    let runtime = Runtime::build_with_llm(config, llm.clone())
        .await
        .expect("runtime starts");

    let tenant = TenantId::new("mock-tenant");
    let project = ProjectId::auto_from_chat("mock-chat");
    wait_for_spec_memory(&data_root, &tenant, &project).await;
    wait_for_llm_calls(&llm, 1).await;

    let user_text = llm
        .last_user_text
        .lock()
        .unwrap()
        .clone()
        .expect("LLM saw a user message");
    assert!(
        user_text.contains("用户上传了以下文件"),
        "default file-only submit should produce attachment summary; got {user_text:?}"
    );
    runtime.shutdown().await;
}

async fn wait_for_spec_memory(data_root: &std::path::Path, tenant: &TenantId, project: &ProjectId) {
    let layout = WorkspaceLayout::new(std::fs::canonicalize(data_root).unwrap()).unwrap();
    let store = MemoryStore::new(layout.memory_dir(tenant, project));
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut names = Vec::new();
    while Instant::now() < deadline {
        names = store.list(MemoryScope::Reference).await.unwrap_or_default();
        if names.iter().any(|n| n.contains("spec")) {
            let stem = names.iter().find(|n| n.contains("spec")).unwrap().clone();
            let entry = store.read(MemoryScope::Reference, &stem).await.unwrap();
            assert!(
                entry.content.contains("kebab-case"),
                "import did not preserve content; got: {:?}",
                entry.content
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("expected `spec`-derived memory entry; got names: {names:?}");
}

async fn wait_for_llm_calls(llm: &ConstantLlm, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if llm.calls.load(Ordering::Relaxed) >= expected {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!(
        "LLM call count stayed below {expected}; got {}",
        llm.calls.load(Ordering::Relaxed)
    );
}

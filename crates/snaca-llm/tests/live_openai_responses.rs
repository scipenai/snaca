//! Live tests for `OpenAiResponsesClient` against an OpenAI Responses
//! endpoint (api.openai.com, or a compatible relay via `OPENAI_BASE_URL`).
//!
//! `live_dashscope.rs` covers a third-party deployment, but it never
//! returns encrypted reasoning, so it can't exercise what only OpenAI's
//! reasoning models do: `include: ["reasoning.encrypted_content"]`,
//! stateless replay of that reasoning across a tool loop, and the
//! non-reasoning models that reject the `include`. This suite does.
//!
//! Skipped by default (`#[ignore]`). Run with:
//!
//! ```bash
//! OPENAI_API_KEY=sk-... \
//! OPENAI_BASE_URL=https://api.openai.com/v1 \  # optional
//! OPENAI_MODEL=gpt-5-mini \                    # a reasoning model
//! OPENAI_CHAT_MODEL=gpt-5-chat-latest \        # a non-reasoning model
//! OPENAI_REASONING_EFFORT=medium \             # optional
//! cargo test -p snaca-llm --test live_openai_responses -- --ignored --nocapture
//! ```

use futures::StreamExt;
use snaca_core::{ContentBlock, ImageSource, Message, Role, ToolUseId};
use snaca_llm::openai_responses::OpenAiResponsesConfig;
use snaca_llm::{
    LlmClient, MessageRequest, MessageResponse, OpenAiResponsesClient, StopReason,
    StreamAccumulator, ToolSchema,
};
use std::sync::{Arc, Mutex};
use tracing::instrument::WithSubscriber;

/// 64x64 solid red PNG (same fixture as `live_dashscope.rs`).
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";

/// The client's log (DEBUG and up) for one test. A rejected replay doesn't
/// fail the call — the client resends without it — and an accepted request
/// doesn't say what it carried, so the replay test reads this to know what
/// was sent. Scoped to one test's future (`with_subscriber`): a global
/// subscriber would mix in the other tests running in parallel.
#[derive(Clone, Default)]
struct ClientLog(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for ClientLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl ClientLog {
    fn subscriber(&self) -> impl tracing::Subscriber + Send + Sync + 'static {
        let sink = self.clone();
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .with_writer(move || sink.clone())
            .finish()
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }

    /// The whitespace-separated `key=value` fields of each request the
    /// client sent, in order.
    fn sent_requests(&self) -> Vec<Vec<String>> {
        self.text()
            .lines()
            .filter(|line| line.contains("sending responses request"))
            .map(|line| line.split_whitespace().map(str::to_string).collect())
            .collect()
    }
}

fn client(model_var: &str, default_model: &str) -> (OpenAiResponsesClient, String) {
    let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env var not set");
    let model = std::env::var(model_var).unwrap_or_else(|_| default_model.into());
    let mut config = OpenAiResponsesConfig::new(key).with_model(&model);
    if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
        config = config.with_base_url(base);
    }
    if let Ok(effort) = std::env::var("OPENAI_REASONING_EFFORT") {
        config = config.with_reasoning_effort(effort);
    }
    (
        OpenAiResponsesClient::new(config).expect("build client"),
        model,
    )
}

fn reasoning_client() -> (OpenAiResponsesClient, String) {
    client("OPENAI_MODEL", "gpt-5-mini")
}

fn text_of(resp: &MessageResponse) -> String {
    resp.message
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_encrypted_reasoning(block: &ContentBlock) -> bool {
    matches!(
        block,
        ContentBlock::Thinking { signature: Some(sig), .. } if sig.starts_with("openai-responses:")
    )
}

fn read_tool() -> ToolSchema {
    ToolSchema {
        name: "Read".into(),
        description: "Read a file from the workspace. Always use it when asked about a file."
            .into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }),
    }
}

fn ask_for_file() -> Message {
    // The call depends on a computation, so the model has to reason before
    // it (a bare "read X" is often answered with zero reasoning tokens, and
    // then there is no reasoning item to keep).
    Message::user_text(
        "Silently work out 37 * 43 — do not write the calculation or its result. If it \
         is odd, use the Read tool on odd.txt, otherwise on even.txt. Then tell me in one \
         sentence what the file says.",
    )
}

/// The assistant turn as the engine would persist it, then the tool result.
fn continue_with_tool_result(first: &MessageResponse, output: &str) -> Vec<Message> {
    let call_id = first
        .message
        .content
        .iter()
        .find_map(|b| match b {
            ContentBlock::ToolUse { id, .. } => Some(id.clone()),
            _ => None,
        })
        .expect("model called the Read tool");
    vec![
        ask_for_file(),
        Message::new(Role::Assistant, first.message.content.clone()),
        Message::new(
            Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new(call_id.as_str()),
                vec![ContentBlock::text(output)],
            )],
        ),
    ]
}

/// Stream `request` and assemble the reply, as the engine does.
async fn streamed(client: &OpenAiResponsesClient, request: MessageRequest) -> MessageResponse {
    let mut stream = client.create_message_stream(request).await.expect("stream");
    let mut acc = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        acc.ingest(event.expect("stream event"));
    }
    acc.finalize().expect("finalize")
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY; live network call"]
async fn streaming_tool_loop_captures_encrypted_reasoning_from_item_done() {
    // Without this: the streaming path (the one the engine uses) could lose
    // the encrypted reasoning — it only arrives on `output_item.done` — or
    // never replay it, and OpenAI accepts the reasoning-less request just
    // the same, so replay would silently never happen in production.
    let (client, model) = reasoning_client();
    let log = ClientLog::default();
    async {
        // Replay keeps reasoning only when it comes before the item it
        // produced (trailing reasoning is trimmed), so the real stream's
        // block order decides whether there is anything to replay. The
        // model sometimes calls the tool without reasoning at all (about
        // one run in six at the default effort), so ask again a few times
        // rather than fail on its choice.
        let reasoned_before_call = |resp: &MessageResponse| {
            let blocks = &resp.message.content;
            let reasoning = blocks.iter().position(is_encrypted_reasoning);
            let call = blocks
                .iter()
                .position(|b| matches!(b, ContentBlock::ToolUse { .. }));
            matches!((reasoning, call), (Some(r), Some(c)) if r < c)
        };
        let mut attempts = Vec::new();
        for _ in 0..3 {
            let reply = streamed(
                &client,
                MessageRequest::new(&model)
                    .with_messages(vec![ask_for_file()])
                    .with_tools(vec![read_tool()]),
            )
            .await;
            assert_eq!(reply.stop_reason, StopReason::ToolUse, "{reply:?}");
            let done = reasoned_before_call(&reply);
            attempts.push(reply);
            if done {
                break;
            }
        }
        let first = attempts.last().expect("at least one attempt");
        assert!(
            reasoned_before_call(first),
            "no encrypted reasoning before the call in {} attempts: {first:?}",
            attempts.len()
        );
        let blocks = &first.message.content;

        let second = streamed(
            &client,
            MessageRequest::new(&model)
                .with_messages(continue_with_tool_result(
                    first,
                    "SNACA is an IM-first agent server.",
                ))
                .with_tools(vec![read_tool()]),
        )
        .await;
        println!("follow-up (stream): {}", text_of(&second));
        assert!(text_of(&second).contains("SNACA"));

        let logged = log.text();
        assert!(
            !logged.contains("encrypted reasoning rejected"),
            "the API rejected the replayed reasoning; the client fell back: {logged}"
        );
        // One request per attempt, then the follow-up. Its input holds one
        // item per block of the first reply (reasoning, message,
        // function_call), plus the ask and the function_call_output: fewer
        // means the reasoning went out without being replayed.
        let requests = log.sent_requests();
        assert_eq!(requests.len(), attempts.len() + 1, "{logged}");
        let follow_up = requests.last().expect("the follow-up request");
        let sent = |field: &str| follow_up.iter().any(|f| f == field);
        assert!(sent("include_encrypted=true"), "{logged}");
        let items = format!("input_items={}", blocks.len() + 2);
        assert!(sent(&items), "expected {items}: {logged}");
    }
    .with_subscriber(log.subscriber())
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY; live network call"]
async fn non_reasoning_model_works_without_include() {
    // Without this: if the model check misfired, every call to a
    // non-reasoning model would 400 on `include`.
    let (client, model) = client("OPENAI_CHAT_MODEL", "gpt-5-chat-latest");
    let resp = client
        .create_message(
            MessageRequest::new(&model).with_messages(vec![Message::user_text(
                "What is 2+2? Answer with the digit.",
            )]),
        )
        .await
        .expect("non-reasoning call");
    let text = text_of(&resp);
    println!("chat model: {text}");
    assert!(text.contains('4'), "{text}");
}

fn recognised_red(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains("red") || text.contains('红')
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY; live network call"]
async fn sees_user_and_tool_returned_images() {
    // Without this: nothing proves the endpoint accepts our `input_image`
    // data URLs, or the follow-up user message that carries tool-returned
    // pixels (tool outputs are text-only on this API).
    let (client, model) = reasoning_client();
    let red = || ContentBlock::Image {
        source: ImageSource::Base64 {
            media_type: "image/png".into(),
            data: RED_PNG_BASE64.into(),
        },
    };
    let direct = client
        .create_message(MessageRequest::new(&model).with_messages(vec![Message::new(
            Role::User,
            vec![
                ContentBlock::text("What colour is this image? One word."),
                red(),
            ],
        )]))
        .await
        .expect("user image");
    println!("user image: {}", text_of(&direct));
    assert!(recognised_red(&text_of(&direct)));

    let via_tool = client
        .create_message(
            MessageRequest::new(&model)
                .with_messages(vec![
                    Message::user_text("Read cat.png and tell me its colour. One word."),
                    Message::new(
                        Role::Assistant,
                        vec![ContentBlock::tool_use(
                            "call_img",
                            "Read",
                            serde_json::json!({"path": "cat.png"}),
                        )],
                    ),
                    Message::new(
                        Role::Tool,
                        vec![ContentBlock::tool_result(
                            ToolUseId::new("call_img"),
                            vec![red()],
                        )],
                    ),
                ])
                .with_tools(vec![read_tool()]),
        )
        .await
        .expect("tool image");
    println!("tool image: {}", text_of(&via_tool));
    assert!(recognised_red(&text_of(&via_tool)));
}

//! HTTP-level integration tests for `OpenAiResponsesClient` using `wiremock`.
//!
//! Verifies request shape and response parsing end-to-end, without hitting
//! a real OpenAI endpoint. Each test stands up its own mock server.

use serde_json::json;
use snaca_core::{ContentBlock, Message, ToolUseId};
use snaca_llm::openai_responses::OpenAiResponsesConfig;
use snaca_llm::{
    ContentBlockStart, ContentDelta, LlmClient, LlmError, MessageRequest, OpenAiResponsesClient,
    StopReason, StreamEvent,
};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config(server: &MockServer) -> OpenAiResponsesConfig {
    OpenAiResponsesConfig::new("test-api-key").with_base_url(server.uri())
}

fn ok_response(output: serde_json::Value) -> serde_json::Value {
    json!({
        "id": "resp_test",
        "model": "gpt-5-mini",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16
        }
    })
}

#[tokio::test]
async fn round_trips_text_request() {
    // Without this: requests could miss /responses, Bearer auth, the
    // `instructions` / `store: false` fields or the input_text shape, and
    // every call to OpenAI would fail (or keep tenant data provider-side).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-api-key"))
        .and(body_partial_json(json!({
            "model": "gpt-5-mini",
            "instructions": "You are SNACA",
            "store": false,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": "hi"}]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hello back"}]}
        ]))))
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    let resp = client
        .create_message(
            MessageRequest::new("gpt-5-mini")
                .with_system("You are SNACA")
                .with_messages(vec![Message::user_text("hi")]),
        )
        .await
        .unwrap();

    assert_eq!(resp.stop_reason, StopReason::EndTurn);
    assert_eq!(resp.usage.input_tokens, 12);
    assert_eq!(resp.usage.output_tokens, 4);
    match &resp.message.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "hello back"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn images_embed_as_input_image_parts() {
    // Without this: the client could default to text-only or mis-shape
    // `input_image`, and OpenAI users' photos would never reach the model.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "input": [{"type": "message", "role": "user",
                       "content": [
                           {"type": "input_text", "text": "look"},
                           {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAEC"}
                       ]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "a cat"}]}
        ]))))
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    let resp = client
        .create_message(
            MessageRequest::new("gpt-5-mini").with_messages(vec![Message::new(
                snaca_core::Role::User,
                vec![
                    ContentBlock::text("look"),
                    ContentBlock::Image {
                        source: snaca_core::ImageSource::Base64 {
                            media_type: "image/png".into(),
                            data: "iVBORw0KGgoAAEC".into(),
                        },
                    },
                ],
            )]),
        )
        .await
        .unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn vision_off_degrades_image_to_placeholder() {
    // Without this: with_vision(false) might not reach the converter, so a
    // text-only deployment is sent image parts and rejects the request.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text",
                                    "text": "[image omitted: image/png — current model has no vision]"}]}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "ok"}]}
        ]))))
        .expect(1)
        .mount(&server)
        .await;

    let client = OpenAiResponsesClient::new(config(&server).with_vision(false)).unwrap();
    client
        .create_message(
            MessageRequest::new("gpt-5-mini").with_messages(vec![Message::new(
                snaca_core::Role::User,
                vec![ContentBlock::Image {
                    source: snaca_core::ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "iVBORw0KGgoAAEC".into(),
                    },
                }],
            )]),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn tool_history_round_trips_through_wire() {
    // Without this: replayed function_call / function_call_output items could
    // lose their call_id pairing or the tool's output, so the API rejects the
    // history or the model never sees what the tool returned. Checked on the
    // recorded request body: `body_partial_json` can't match subarrays.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "done"}]}
        ]))))
        .expect(1)
        .mount(&server)
        .await;

    let history = vec![
        Message::user_text("read README"),
        Message::new(
            snaca_core::Role::Assistant,
            vec![ContentBlock::tool_use(
                "call_1",
                "Read",
                json!({"path": "README.md"}),
            )],
        ),
        Message::new(
            snaca_core::Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new("call_1"),
                vec![ContentBlock::text("# Hello")],
            )],
        ),
        Message::user_text("now summarise"),
    ];

    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    let resp = client
        .create_message(MessageRequest::new("gpt-5-mini").with_messages(history))
        .await
        .unwrap();
    assert_eq!(resp.stop_reason, StopReason::EndTurn);

    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let input = body["input"].as_array().unwrap();
    assert!(input.iter().any(|item| {
        item["type"] == "function_call" && item["call_id"] == "call_1" && item["name"] == "Read"
    }));
    assert!(input.iter().any(|item| {
        item["type"] == "function_call_output"
            && item["call_id"] == "call_1"
            && item["output"] == "# Hello"
    }));
}

#[tokio::test]
async fn streaming_sse_yields_canonical_events() {
    // Without this: the stream call could go out without `stream: true` or
    // bypass the SSE parser, and every engine turn (always streamed) would
    // end as an interrupted stream.
    let sse = "\
event: response.created\n\
data: {\"response\":{\"id\":\"resp_sse\",\"model\":\"gpt-5-mini\"}}\n\
\n\
event: response.output_text.delta\n\
data: {\"output_index\":0,\"delta\":\"Hello\"}\n\
\n\
event: response.output_item.added\n\
data: {\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"Bash\",\"arguments\":\"\"}}\n\
\n\
event: response.function_call_arguments.delta\n\
data: {\"output_index\":1,\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\
\n\
event: response.completed\n\
data: {\"response\":{\"id\":\"resp_sse\",\"status\":\"completed\",\"usage\":{\"input_tokens\":9,\"output_tokens\":5}}}\n\
\n";

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;

    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    use futures::StreamExt;
    let mut stream = client
        .create_message_stream(
            MessageRequest::new("gpt-5-mini").with_messages(vec![Message::user_text("hi")]),
        )
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }

    assert!(matches!(
        &events[0],
        StreamEvent::MessageStart { message_id, .. } if message_id == "resp_sse"
    ));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::ContentBlockStart {
            block: ContentBlockStart::ToolUse { id, name },
            ..
        } if id == "call_9" && name == "Bash"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::ContentBlockDelta {
            delta: ContentDelta::Text { text },
            ..
        } if text == "Hello"
    )));
    match events.iter().find_map(|e| match e {
        StreamEvent::MessageDelta { stop_reason, usage } => {
            Some((stop_reason.clone(), usage.clone()))
        }
        _ => None,
    }) {
        Some((Some(StopReason::ToolUse), Some(usage))) => {
            assert_eq!(usage.input_tokens, 9);
        }
        other => panic!("expected ToolUse delta with usage, got {other:?}"),
    }
    assert!(matches!(events.last(), Some(StreamEvent::MessageStop)));
}

#[tokio::test]
async fn rate_limit_maps_to_rate_limited_with_retry_after() {
    // Without this: a 429's Retry-After could be dropped, so the retry
    // wrapper backs off blindly instead of waiting as long as OpenAI asked.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "7")
                .set_body_json(json!({
                    "error": {
                        "message": "rate limit exceeded",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit"
                    }
                })),
        )
        .mount(&server)
        .await;
    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    let err = client
        .create_message(
            MessageRequest::new("gpt-5-mini").with_messages(vec![Message::user_text("hi")]),
        )
        .await
        .unwrap_err();
    match err {
        LlmError::RateLimited { retry_after } => {
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(7)));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn server_5xx_maps_to_server_transient() {
    // Without this: a plain-text or HTML 5xx from a gateway could surface as
    // a parse error instead of ServerTransient, and the turn fails unretried.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
        .mount(&server)
        .await;
    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    let err = client
        .create_message(
            MessageRequest::new("gpt-5-mini").with_messages(vec![Message::user_text("hi")]),
        )
        .await
        .unwrap_err();
    match err {
        LlmError::ServerTransient { status } => assert_eq!(status, 500),
        other => panic!("expected ServerTransient, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_api_key_rejected_at_construction() {
    // Without this: an empty API key (e.g. OPENAI_API_KEY set to "") is
    // accepted at construction and only fails as a 401 on the first message.
    match OpenAiResponsesClient::new(OpenAiResponsesConfig::new("")) {
        Err(LlmError::InvalidConfig(_)) => {}
        Err(other) => panic!("expected InvalidConfig, got {other:?}"),
        Ok(_) => panic!("expected construction to fail with empty api key"),
    }
}

#[tokio::test]
async fn reasoning_model_requests_encrypted_reasoning_and_retries_without_it_when_rejected() {
    // Without this: (1) OpenAI returns no encrypted reasoning unless asked,
    // so replay never happens on gpt-5 / o-series; (2) a replayed blob the
    // API can't verify (`invalid_encrypted_content`) fails the whole turn
    // instead of degrading to no replay.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"error": {
            "type": "invalid_request_error",
            "code": "invalid_encrypted_content",
            "message": "The encrypted content could not be verified."}})))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "done"}]}
        ]))))
        .mount(&server)
        .await;

    let history = vec![
        Message::user_text("read x"),
        Message::new(
            snaca_core::Role::Assistant,
            vec![
                ContentBlock::Thinking {
                    text: "plan".into(),
                    signature: Some("openai-responses:gAAA".into()),
                },
                ContentBlock::tool_use("call_1", "Read", json!({"path": "x"})),
            ],
        ),
        Message::new(
            snaca_core::Role::Tool,
            vec![ContentBlock::tool_result(
                ToolUseId::new("call_1"),
                vec![ContentBlock::text("ok")],
            )],
        ),
    ];
    let client = OpenAiResponsesClient::new(config(&server).with_model("gpt-5-mini")).unwrap();
    let resp = client
        .create_message(MessageRequest::new("gpt-5-mini").with_messages(history))
        .await
        .expect("retry without replay succeeds");
    assert!(matches!(&resp.message.content[0], ContentBlock::Text { text } if text == "done"));

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let first: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    let second: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    let has_reasoning = |body: &serde_json::Value| {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == "reasoning")
    };
    assert_eq!(first["include"], json!(["reasoning.encrypted_content"]));
    assert!(has_reasoning(&first));
    assert!(second.get("include").is_none(), "{second}");
    assert!(!has_reasoning(&second));
}

#[tokio::test]
async fn non_reasoning_model_never_sends_include() {
    // Without this: gpt-4o / gpt-4.1 / *-chat reject
    // `include: ["reasoning.encrypted_content"]` with a 400 on every call.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(ok_response(json!([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hi"}]}
        ]))))
        .mount(&server)
        .await;
    let client = OpenAiResponsesClient::new(config(&server)).unwrap();
    client
        .create_message(
            MessageRequest::new("gpt-4.1").with_messages(vec![Message::user_text("hi")]),
        )
        .await
        .unwrap();
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert!(body.get("include").is_none(), "{body}");
}

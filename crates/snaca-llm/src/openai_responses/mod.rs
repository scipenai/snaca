//! OpenAI Responses API client (`POST {base}/responses`).
//!
//! The Responses API is OpenAI's successor to Chat Completions and the
//! native protocol for the GPT-5 / o-series family. Compared to the
//! DeepSeek client (OpenAI Chat Completions compatible):
//!
//! - System prompt is a top-level `instructions` field, not a message.
//! - Tool calls are top-level `function_call` / `function_call_output`
//!   *items* in the `input` array, not `tool_calls` on a message.
//! - Images are `input_image` content parts (data URLs allowed).
//! - `store: false` is always sent: history is fully replayed from
//!   SQLite each turn and multi-tenant IM content must not be retained
//!   provider-side.
//! - Reasoning is kept statelessly as `Thinking` blocks. Their text is the
//!   reasoning summary when the endpoint sends one; OpenAI only does when
//!   `reasoning.summary` is requested, which SNACA doesn't since it can
//!   require organization verification, so there the text is empty. The
//!   reasoning item's `encrypted_content` is stored as the block's
//!   signature and, within the current tool loop, replayed as id-less
//!   `reasoning` items, as OpenAI asks for reasoning models doing
//!   function calls. `include: ["reasoning.encrypted_content"]` — what
//!   makes the API return it — is sent only for reasoning models
//!   ([`is_reasoning_model`], or [`OpenAiResponsesConfig::with_encrypted_reasoning`]):
//!   non-reasoning models reject it with a 400. If the API rejects a
//!   replayed blob (`invalid_encrypted_content`), the request is resent
//!   once without replay.
//!
//! Auth uses `Authorization: Bearer`. See
//! <https://platform.openai.com/docs/api-reference/responses> for
//! protocol details.

mod convert;
mod sse;
mod wire;

/// Marks a `Thinking::signature` as this provider's encrypted reasoning.
/// Replay here only accepts blobs carrying it, and no other converter
/// sends a signature carrying it (Anthropic drops the block, DeepSeek
/// replays only its text) — a thread that switched providers must never
/// send one provider's opaque token to another.
pub(crate) const REASONING_SIGNATURE_PREFIX: &str = "openai-responses:";

use crate::client::{LlmClient, ProviderCaps};
use crate::error::{LlmError, LlmResult};
use crate::request::MessageRequest;
use crate::response::MessageResponse;
use crate::stream::StreamEvent;
use crate::transport::{
    classify_error, log_response_headers, retry_after_header, wrap_byte_stream,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::time::Duration;
use tracing::{debug, warn};
use wire::{ResponsesResponse, WireErrorEnvelope};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-5-mini";
/// Max idle gap between SSE chunks (and total wait for a non-streaming
/// response body). Same reasoning as the other providers: reasoning
/// models legitimately stream for minutes, so this is a `read_timeout`
/// rather than a total request budget.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct OpenAiResponsesConfig {
    pub api_key: String,
    pub base_url: String,
    pub default_model: String,
    pub request_timeout: Duration,
    /// Embed images as `input_image` content parts. Default `true` —
    /// the Responses API's flagship models are multimodal. Flip to
    /// `false` for text-only deployments or when pointing `base_url`
    /// at a vision-less compatible endpoint.
    pub supports_vision: bool,
    /// Request `reasoning.encrypted_content` so the stateless tool loop can
    /// replay reasoning. `None` decides per model ([`is_reasoning_model`]):
    /// non-reasoning models reject the `include` with a 400.
    pub encrypted_reasoning: Option<bool>,
    /// `reasoning.effort` to request (`low`, `medium`, `high`, …). `None`
    /// leaves it to the endpoint's default — which on some deployments means
    /// little or no reasoning once tools are offered.
    pub reasoning_effort: Option<String>,
}

impl OpenAiResponsesConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.into(),
            default_model: DEFAULT_MODEL.into(),
            request_timeout: DEFAULT_TIMEOUT,
            supports_vision: true,
            encrypted_reasoning: None,
            reasoning_effort: None,
        }
    }

    /// API root including the version segment, as in the OpenAI SDKs
    /// (`https://api.openai.com/v1`): requests go to `{base}/responses`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.request_timeout = t;
        self
    }

    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.supports_vision = enabled;
        self
    }

    /// Force `include: ["reasoning.encrypted_content"]` on or off, e.g. for
    /// a reasoning deployment whose name the model check doesn't recognise.
    pub fn with_encrypted_reasoning(mut self, enabled: bool) -> Self {
        self.encrypted_reasoning = Some(enabled);
        self
    }

    /// Request a reasoning effort (`low`, `medium`, `high`, …) instead of
    /// the endpoint's default. Only meaningful for reasoning models.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }
}

/// Whether `model` names an OpenAI reasoning model — the ones that return
/// encrypted reasoning: GPT-5 and every later major version (`gpt-5.4-mini`,
/// `gpt-6.1-sol`, …) and the o-series. `chat` variants (`gpt-5-chat-latest`)
/// are non-reasoning. A vendor prefix (`openai/…`) is ignored.
pub fn is_reasoning_model(model: &str) -> bool {
    let name = model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase();
    if name.contains("chat") {
        return false;
    }
    let gpt_major = name
        .strip_prefix("gpt-")
        .map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .and_then(|digits| digits.parse::<u32>().ok());
    gpt_major.is_some_and(|major| major >= 5)
        || (name.starts_with('o') && name[1..].starts_with(|c: char| c.is_ascii_digit()))
}

#[derive(Clone)]
pub struct OpenAiResponsesClient {
    config: OpenAiResponsesConfig,
    http: reqwest::Client,
}

impl OpenAiResponsesClient {
    pub fn new(config: OpenAiResponsesConfig) -> LlmResult<Self> {
        if config.api_key.is_empty() {
            return Err(LlmError::InvalidConfig("api_key is empty".into()));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .read_timeout(config.request_timeout)
            .user_agent(concat!("snaca-llm/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { config, http })
    }

    /// Construct a client pointing at an arbitrary base URL — used by tests
    /// against a mock HTTP server.
    pub fn with_http_client(config: OpenAiResponsesConfig, http: reqwest::Client) -> Self {
        Self { config, http }
    }

    /// Send `request`; if the API rejects replayed encrypted reasoning
    /// (`invalid_encrypted_content` — e.g. it was produced under different
    /// keys or settings), resend once without reasoning replay or
    /// `include` rather than failing the turn.
    async fn post(&self, request: &MessageRequest, stream: bool) -> LlmResult<reqwest::Response> {
        match self.post_once(request, stream, true).await {
            Err(LlmError::Provider { code, message }) if code == "invalid_encrypted_content" => {
                warn!(
                    provider = "openai-responses",
                    %message,
                    "encrypted reasoning rejected; retrying without reasoning replay"
                );
                self.post_once(request, stream, false).await
            }
            other => other,
        }
    }

    async fn post_once(
        &self,
        request: &MessageRequest,
        stream: bool,
        replay: bool,
    ) -> LlmResult<reqwest::Response> {
        let include_encrypted = replay
            && self
                .config
                .encrypted_reasoning
                .unwrap_or_else(|| is_reasoning_model(&request.model));
        let body = convert::build_responses_request_with(
            request,
            stream,
            self.config.supports_vision,
            convert::ReasoningMode {
                include_encrypted,
                replay,
                effort: self.config.reasoning_effort.clone(),
            },
        )?;
        debug!(
            provider = "openai-responses",
            model = %body.model,
            input_items = body.input.len(),
            tools = body.tools.len(),
            stream,
            include_encrypted,
            "sending responses request"
        );
        let mut builder = self
            .http
            .post(self.endpoint())
            .bearer_auth(&self.config.api_key);
        if stream {
            builder = builder.header("accept", "text/event-stream");
        }
        let resp = builder.json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let retry_after = retry_after_header(&resp);
            let bytes = resp.bytes().await?;
            return Err(classify_error::<WireErrorEnvelope>(
                status.as_u16(),
                retry_after.as_deref(),
                &bytes,
            ));
        }
        Ok(resp)
    }

    fn endpoint(&self) -> String {
        format!("{}/responses", self.config.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl LlmClient for OpenAiResponsesClient {
    fn provider_name(&self) -> &'static str {
        "openai-responses"
    }

    fn model(&self) -> &str {
        &self.config.default_model
    }

    fn capabilities(&self) -> ProviderCaps {
        ProviderCaps {
            tool_use: true,
            // OpenAI caches long prompts transparently (no cache_control
            // knob), like DeepSeek.
            prompt_cache: true,
            // Reasoning summaries surface as canonical Thinking blocks;
            // encrypted reasoning is replayed within the tool loop (see
            // module docs).
            thinking: true,
            streaming: true,
        }
    }

    async fn create_message(&self, mut request: MessageRequest) -> LlmResult<MessageResponse> {
        if request.model.is_empty() {
            request.model = self.config.default_model.clone();
        }
        let resp = self.post(&request, false).await?;
        let bytes = resp.bytes().await?;
        let parsed: ResponsesResponse = serde_json::from_slice(&bytes).map_err(|e| {
            warn!(error = %e, body = %String::from_utf8_lossy(&bytes), "failed to parse responses response");
            LlmError::MalformedResponse(format!("failed to deserialise response: {e}"))
        })?;
        convert::parse_responses_response(parsed)
    }

    async fn create_message_stream(
        &self,
        mut request: MessageRequest,
    ) -> LlmResult<BoxStream<'static, LlmResult<StreamEvent>>> {
        if request.model.is_empty() {
            request.model = self.config.default_model.clone();
        }
        let resp = self.post(&request, true).await?;
        log_response_headers("openai-responses", &resp);
        let byte_stream = wrap_byte_stream("openai-responses", resp.bytes_stream());
        Ok(sse::parse_byte_stream(byte_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::is_reasoning_model;

    #[test]
    fn reasoning_models_are_recognised_by_name() {
        // Without this: a wrong guess either 400s every call of a
        // non-reasoning model (include sent) or silently disables replay
        // for a reasoning one (include withheld).
        for model in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1-codex",
            "gpt-6.1-sol",
            "gpt-10",
            "o1",
            "o3-mini",
            "o4-mini",
            "openai/gpt-5",
        ] {
            assert!(is_reasoning_model(model), "{model}");
        }
        for model in [
            "gpt-4o",
            "gpt-4.1",
            "gpt-5-chat-latest",
            "gpt-5.2-chat-latest",
            "gpt-image-1",
            "omni-moderation",
            "qwen3.8-flash",
        ] {
            assert!(!is_reasoning_model(model), "{model}");
        }
    }
}

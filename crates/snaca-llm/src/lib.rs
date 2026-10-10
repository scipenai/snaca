//! `snaca-llm` — LLM provider abstraction.
//!
//! Defines a provider-agnostic [`LlmClient`] trait, the canonical
//! request/response shapes, and concrete implementations.
//!
//! M1: DeepSeek (OpenAI-compatible, non-stream first; streaming in M2).
//! Anthropic native is stubbed; full implementation lands in M2.
//!
//! ## Layout
//! - [`error`]    — `LlmError` / `LlmResult`
//! - [`request`]  — `MessageRequest` + `ToolSchema` + `StopSequence`
//! - [`response`] — `MessageResponse` + `StopReason`
//! - [`client`]   — `LlmClient` trait + `ProviderCaps`
//! - [`deepseek`] — DeepSeek client (OpenAI Chat Completions compatible)
//! - [`anthropic`] — Anthropic Messages API client
//! - [`openai_responses`] — OpenAI Responses API client (GPT-5 / o-series)
//! - `images` — shared image-block helpers for provider converters (crate-private)

pub mod anthropic;
pub mod classify;
pub mod client;
pub mod deepseek;
pub mod error;
mod images;
pub mod openai_responses;
pub mod request;
pub mod response;
pub mod retry;
pub mod stream;
pub(crate) mod transport;

pub use anthropic::{AnthropicClient, AnthropicConfig};
pub use classify::classify_http_error;
pub use client::{LlmClient, ProviderCaps};
pub use deepseek::{DeepSeekClient, DeepSeekConfig};
pub use error::{LlmError, LlmResult};
pub use openai_responses::{OpenAiResponsesClient, OpenAiResponsesConfig};
pub use request::{MessageRequest, SystemSegment, ToolSchema};
pub use response::{MessageResponse, StopReason};
pub use retry::{RetryConfig, RetryingLlmClient};
pub use stream::{
    synthesize_events, ContentBlockStart, ContentDelta, StreamAccumulator, StreamEvent,
};

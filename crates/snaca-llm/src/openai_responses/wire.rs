//! OpenAI Responses API wire types — `POST {base}/responses`.
//!
//! The engine never sees these — `convert.rs` translates to/from the
//! provider-agnostic `MessageRequest` / `MessageResponse`. Shapes mirror
//! the public API reference, limited to the fields we send or read; a few
//! deserialised fields go unread, so `dead_code` is suppressed at the
//! module level rather than per field.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------- request ----------------

#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    /// Flattened system prompt (top-level + history `Role::System`
    /// messages). The Responses API has no per-segment cache knob —
    /// caching is automatic server-side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<InputItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    /// Always `false` (see module docs on [`crate::openai_responses`]).
    pub store: bool,
    /// `["reasoning.encrypted_content"]` for reasoning models, so stateless
    /// responses carry reasoning that can be replayed; omitted otherwise
    /// (non-reasoning models reject it with a 400).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<&'static str>,
    /// Reasoning effort (`minimal` / `low` / `medium` / `high` / …), sent
    /// only when configured; otherwise the endpoint's default applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningParam>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningParam {
    pub effort: String,
}

/// One item of the `input` array. Unlike Chat Completions, tool calls
/// are sibling items of messages, not fields on an assistant message.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message {
        role: String,
        content: Vec<InputContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        /// JSON-encoded arguments string — the OpenAI function-tool
        /// convention, same as Chat Completions.
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        /// Text only. Tool-returned images are re-surfaced as a follow-up
        /// user message instead (see convert): compatible gateways may not
        /// take image parts here, as OpenAI itself does.
        output: String,
    },
    /// Replayed encrypted reasoning (stateless, `store: false`). Sent
    /// without an `id`: an id makes the API look the item up server-side,
    /// which fails when nothing is stored, and would also require the
    /// following `function_call` / `message` items to carry their own
    /// ids. Must sit immediately before the item it produced.
    Reasoning {
        summary: Vec<InputSummaryPart>,
        encrypted_content: String,
    },
}

/// One reasoning-summary part on a replayed `reasoning` item.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputSummaryPart {
    SummaryText { text: String },
}

/// Content parts inside an input `message` item. User messages use
/// `input_text` / `input_image`; replayed assistant messages use
/// `output_text` (the API's own output shape echoed back).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    InputText {
        text: String,
    },
    InputImage {
        /// Data URL (`data:<media>;base64,<data>`) or a public https URL.
        image_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    OutputText {
        text: String,
    },
}

/// Responses function tools are flat — `name` / `parameters` live on the
/// tool itself, not nested under a `function` object like Chat
/// Completions. `strict` is always `false`: SNACA's tool schemas are
/// hand-built JSON Schema and don't satisfy strict-mode constraints.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub strict: bool,
}

// ---------------- response ----------------

#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    /// `completed` | `failed` | `incomplete` | `in_progress` | `queued`.
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
    #[serde(default)]
    pub error: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// One item of the `output` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message {
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        content: Vec<OutputContentPart>,
    },
    FunctionCall {
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        arguments: Option<String>,
    },
    /// Reasoning. `encrypted_content` is present when the request asked for
    /// it (`include`, sent for reasoning models); convert stores it on the canonical
    /// `Thinking` block so the next request can replay it.
    Reasoning {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<OutputSummaryPart>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputSummaryPart {
    SummaryText {
        text: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct WireUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<Value>,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

impl From<WireUsage> for snaca_core::Usage {
    fn from(w: WireUsage) -> Self {
        let cached = w.input_tokens_details.map(|d| d.cached_tokens).unwrap_or(0);
        snaca_core::Usage {
            input_tokens: w.input_tokens,
            output_tokens: w.output_tokens,
            cache_read_input_tokens: (cached > 0).then_some(cached),
            cache_creation_input_tokens: None,
        }
    }
}

// ---------------- errors ----------------

/// OpenAI's standard error envelope (`{error: {message, type, code}}`) is
/// shared with Chat Completions, so reuse the DeepSeek client's definition.
pub(crate) use crate::deepseek::wire::WireErrorEnvelope;

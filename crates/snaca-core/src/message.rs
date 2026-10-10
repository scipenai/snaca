//! Conversation messages.
//!
//! [`Message`] is the canonical conversation unit — a [`Role`] plus an ordered
//! list of [`crate::ContentBlock`]s. Stored as-is in `snaca-state`; LLM
//! providers convert into provider-native shapes at the boundary.

use crate::content::ContentBlock;
use crate::ids::MessageId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Synthetic role for tool result envelopes — Anthropic packages tool
    /// results as user-role messages, but we keep them logically tagged so
    /// the engine can route them correctly.
    Tool,
}

/// Lead-in of the corrective user messages the engine injects into a tool
/// loop (malformed tool arguments, a repeated identical failure). They are
/// `Role::User` because every provider accepts that mid-loop, but they
/// continue the turn rather than start one — see [`current_turn_start`].
pub const ENGINE_FEEDBACK_PREFIX: &str = "[SNACA FEEDBACK] ";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: MessageId,
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub created_at: DateTime<Utc>,
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self {
            id: MessageId::new(),
            role,
            content,
            created_at: Utc::now(),
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    /// A user message that starts a turn — any but the engine's mid-loop
    /// feedback ([`ENGINE_FEEDBACK_PREFIX`]).
    pub fn starts_turn(&self) -> bool {
        matches!(self.role, Role::User)
            && !matches!(
                self.content.first(),
                Some(ContentBlock::Text { text }) if text.starts_with(ENGINE_FEEDBACK_PREFIX)
            )
    }
}

/// Index of the first message after the last one that starts a turn: from
/// here on is the current turn's tool loop, whose images and reasoning the
/// model still needs. `0` when no message starts a turn. Engine feedback
/// injected mid-loop doesn't move it; the compaction summary does, since
/// the loop before it has been summarised away.
pub fn current_turn_start(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(Message::starts_turn)
        .map_or(0, |i| i + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_has_text_content() {
        let m = Message::user_text("hi");
        assert!(matches!(m.role, Role::User));
        assert_eq!(m.content.len(), 1);
        assert!(matches!(&m.content[0], ContentBlock::Text { text } if text == "hi"));
    }

    #[test]
    fn engine_feedback_does_not_start_a_turn() {
        // Without this: retry feedback injected mid-loop would count as the
        // turn start, so images and reasoning from the loop before it would
        // be retired / not replayed while the model is still working on it.
        let feedback = format!("{ENGINE_FEEDBACK_PREFIX}the JSON arguments could not be parsed");
        let msgs = vec![
            Message::user_text("compare these photos"),
            Message::assistant_text("calling Read"),
            Message::user_text(feedback),
            Message::assistant_text("calling Read again"),
        ];
        assert_eq!(current_turn_start(&msgs), 1);
        assert!(!Message::assistant_text("hi").starts_turn());
    }

    #[test]
    fn compaction_summary_bounds_the_current_turn() {
        // Without this: when compaction swallows the current request, the
        // preserved head's tail would count as the current loop, and its
        // images would never be retired.
        let msgs = vec![
            Message::user_text("first request"),
            Message::assistant_text("calling Read"),
            Message::user_text("[SNACA SUMMARY of earlier conversation]"),
            Message::assistant_text("calling Read again"),
        ];
        assert_eq!(current_turn_start(&msgs), 3);
        assert_eq!(current_turn_start(&[Message::assistant_text("ok")]), 0);
    }

    #[test]
    fn role_serialises_lowercase() {
        let json = serde_json::to_string(&Role::Assistant).unwrap();
        assert_eq!(json, "\"assistant\"");
    }
}

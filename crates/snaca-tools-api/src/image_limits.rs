//! `ImageLimits` — the budget a tool normalises captured images to.
//!
//! Lives here (not in `snaca-tools`) because three layers share it: the
//! host's config validation, the engine that attaches it to every
//! [`crate::ToolContext`], and the tools that read it back. The bounds are
//! part of the type so the server can reject an out-of-range config and a
//! tool can still clamp defensively when an SDK embedder sets it by hand.
//!
//! Byte budgets are measured on the **base64** payload — the size an image
//! actually occupies in serialized history and on the wire, and the same
//! unit the engine's `history_max_bytes` / `max_tool_result_bytes`
//! accounting uses. (hermes's `vision.embed_target_bytes` is measured the
//! same way, so equal settings mean equal footprints.)

use snaca_core::ImageSource;

/// Per-image budget applied when a tool returns pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLimits {
    /// Base64-size ceiling for one embedded image after normalisation
    /// (raw bytes × 4/3, rounded up to whole 4-byte groups).
    pub max_bytes: usize,
    /// Longest-edge ceiling in pixels.
    pub max_edge_px: u32,
}

impl ImageLimits {
    /// Default budget (≈192 KB of raw image): several images fit under
    /// the engine's default 1.5 MB history cap.
    pub const DEFAULT_MAX_BYTES: usize = 256 * 1024;
    /// Default edge cap. Anthropic downsamples anything larger anyway.
    pub const DEFAULT_MAX_EDGE_PX: u32 = 1568;

    /// Below this the JPEG ladder can't produce a legible image.
    pub const MIN_BYTES: usize = 64 * 1024;
    /// The provider-side per-image cap the `snaca-llm` send-time gate
    /// enforces. Anything larger would be captured only to be refused at
    /// the wire.
    pub const MAX_BYTES: usize = ImageSource::MAX_INLINE_BASE64_BYTES;
    /// Below this an image stops being useful to look at.
    pub const MIN_EDGE_PX: u32 = 256;
    /// Anthropic's hard per-image dimension limit.
    pub const MAX_EDGE_PX: u32 = 8000;

    /// Both fields pulled into their allowed ranges.
    pub fn clamped(self) -> Self {
        Self {
            max_bytes: self.max_bytes.clamp(Self::MIN_BYTES, Self::MAX_BYTES),
            max_edge_px: self.max_edge_px.clamp(Self::MIN_EDGE_PX, Self::MAX_EDGE_PX),
        }
    }
}

impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_bytes: Self::DEFAULT_MAX_BYTES,
            max_edge_px: Self::DEFAULT_MAX_EDGE_PX,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamped_pulls_fields_into_range_and_keeps_valid_values() {
        // Without this: an SDK-built context can push the edge cap past 8000 px
        // (images Anthropic rejects) or the byte budget past the 5 MB the
        // send-time gate refuses, so Read captures images the wire won't take.
        let low_bytes_high_edge = ImageLimits {
            max_bytes: 1,
            max_edge_px: u32::MAX,
        };
        assert_eq!(
            low_bytes_high_edge.clamped(),
            ImageLimits {
                max_bytes: ImageLimits::MIN_BYTES,
                max_edge_px: ImageLimits::MAX_EDGE_PX,
            }
        );
        let high_bytes_low_edge = ImageLimits {
            max_bytes: usize::MAX,
            max_edge_px: 0,
        };
        assert_eq!(
            high_bytes_low_edge.clamped(),
            ImageLimits {
                max_bytes: ImageLimits::MAX_BYTES,
                max_edge_px: ImageLimits::MIN_EDGE_PX,
            }
        );
        assert_eq!(ImageLimits::default().clamped(), ImageLimits::default());
    }
}

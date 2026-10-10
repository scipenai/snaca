//! Shared image-block helpers used by every provider converter.
//!
//! `snaca_core::ImageSource` is the canonical image shape; every provider
//! needs the same two derivations from it — a URL string (data URL for
//! base64, passthrough for plain URLs) and a label for the no-vision
//! placeholder. Kept in one place so the placeholder wording stays
//! identical across providers and the wire shapes can't drift apart.
//!
//! [`omitted_placeholder`] is the last gate before pixels hit the wire. The Read
//! tool already normalises what it captures, but history can hold images
//! from older builds or other producers; a format or size the provider
//! rejects would 400 the request — and because the block lives in
//! history, every later request in the thread too. Anything the gate
//! refuses degrades to placeholder text instead.

use data_encoding::BASE64;
use snaca_core::{ContentBlock, ImageSource};

/// What a provider's wire accepts as inline images.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImagePolicy {
    /// The model can see images at all.
    pub vision: bool,
    /// Only non-animated GIFs are accepted.
    pub static_gif_only: bool,
}

impl ImagePolicy {
    /// Anthropic Messages: animated GIFs are accepted.
    pub(crate) fn anthropic(vision: bool) -> Self {
        Self {
            vision,
            static_gif_only: false,
        }
    }

    /// OpenAI Chat Completions / Responses (and compatible endpoints):
    /// "non-animated GIF" only.
    pub(crate) fn openai(vision: bool) -> Self {
        Self {
            vision,
            static_gif_only: true,
        }
    }
}

/// Why an image can't go on the wire as pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    NoVision,
    UnsupportedFormat,
    TooLarge,
    MediaTypeMismatch,
    AnimatedGif,
}

/// The whitelisted format the payload's magic bytes identify, read straight
/// off the base64 prefix (each signature encodes to fixed characters, so
/// nothing is decoded). `None` for anything unrecognised.
fn sniffed_media_type(data: &str) -> Option<&'static str> {
    if data.starts_with("iVBORw0KGgo") {
        Some("image/png") // \x89PNG\r\n\x1a\n
    } else if data.starts_with("/9j/") {
        Some("image/jpeg") // FF D8 FF
    } else if data.starts_with("R0lGOD") {
        Some("image/gif") // GIF8
    } else if data.starts_with("UklGR") && data.get(12..16) == Some("RUJQ") {
        Some("image/webp") // RIFF....WEBP
    } else {
        None
    }
}

/// More than one frame in a base64 GIF. Walks the block structure
/// (extensions and image data are length-prefixed sub-blocks), counting
/// image descriptors; unparseable data is not second-guessed.
fn is_animated_gif(data: &str) -> bool {
    let Ok(bytes) = BASE64.decode(data.as_bytes()) else {
        return false;
    };
    gif_frame_count(&bytes, 2).is_some_and(|frames| frames >= 2)
}

/// Image descriptors in `bytes`, stopping once `limit` is reached. `None`
/// if the data isn't a well-formed GIF up to that point.
fn gif_frame_count(bytes: &[u8], limit: usize) -> Option<usize> {
    // Size of a colour table flagged in `packed`: 3 · 2^(n+1) bytes.
    let color_table = |packed: u8| {
        if packed & 0x80 != 0 {
            3 << ((packed & 0x07) + 1)
        } else {
            0
        }
    };
    // A run of length-prefixed sub-blocks, ended by a zero-length one.
    let skip_sub_blocks = |mut pos: usize| -> Option<usize> {
        loop {
            let len = *bytes.get(pos)? as usize;
            pos += 1 + len;
            if len == 0 {
                return Some(pos);
            }
        }
    };
    if !bytes.starts_with(b"GIF8") {
        return None;
    }
    let mut pos = 13 + color_table(*bytes.get(10)?); // header + screen descriptor
    let mut frames = 0;
    while frames < limit {
        match *bytes.get(pos)? {
            0x21 => pos = skip_sub_blocks(pos + 2)?, // extension: introducer, label
            0x2C => {
                frames += 1;
                if frames == limit {
                    // Counted; whatever follows (even cut off) can't undo it.
                    break;
                }
                // Descriptor (10 bytes), local colour table, LZW min code size.
                pos += 10 + color_table(*bytes.get(pos + 9)?) + 1;
                pos = skip_sub_blocks(pos)?;
            }
            0x3B => break, // trailer
            _ => return None,
        }
    }
    Some(frames)
}

fn refusal(source: &ImageSource, policy: ImagePolicy) -> Option<Refusal> {
    if !policy.vision {
        return Some(Refusal::NoVision);
    }
    match source {
        // The provider fetches URLs itself; nothing to check locally.
        ImageSource::Url { .. } => None,
        ImageSource::Base64 { media_type, data } => {
            // Exact match: the string is sent as-is, and providers compare
            // it exactly — `IMAGE/PNG` would pass a lenient check and 400.
            if !ImageSource::INLINE_MEDIA_TYPES.contains(&media_type.as_str()) {
                Some(Refusal::UnsupportedFormat)
            } else if data.len() > ImageSource::MAX_INLINE_BASE64_BYTES {
                Some(Refusal::TooLarge)
            } else if sniffed_media_type(data) != Some(media_type.as_str()) {
                // The bytes must be the labelled format. History written
                // before Read sniffed content was labelled by extension, so
                // a `.png` can hold a JPEG — or no image at all, such as a
                // saved HTML error page. Providers check the bytes against
                // the label and 400 on every replay.
                Some(Refusal::MediaTypeMismatch)
            } else if policy.static_gif_only && media_type == "image/gif" && is_animated_gif(data) {
                // Read reduces animations to one frame, but history written
                // before that — or a custom tool's output — can still hold
                // one, and the provider would reject it on every replay.
                Some(Refusal::AnimatedGif)
            } else {
                None
            }
        }
    }
}

/// Base64 images become `data:` URLs; plain URLs pass through unchanged —
/// both are accepted shapes for OpenAI `image_url` / `input_image` parts.
pub(crate) fn image_url_of(source: &ImageSource) -> String {
    match source {
        ImageSource::Url { url } => url.clone(),
        ImageSource::Base64 { media_type, data } => format!("data:{media_type};base64,{data}"),
    }
}

/// `None` when `source` can be sent as an image part; otherwise the
/// one-line text to send in its place, so the model at least knows an
/// image was there, which media type, and why it can't see it, instead
/// of the bytes silently vanishing. One call per image: the checks
/// include decoding a GIF's frame structure.
pub(crate) fn omitted_placeholder(source: &ImageSource, policy: ImagePolicy) -> Option<String> {
    let reason = match refusal(source, policy)? {
        Refusal::UnsupportedFormat => "unsupported image format",
        Refusal::TooLarge => "image exceeds the provider size limit",
        Refusal::MediaTypeMismatch => "media type does not match the image data",
        Refusal::AnimatedGif => "animated GIF; this provider accepts static images only",
        Refusal::NoVision => "current model has no vision",
    };
    let label = match source {
        ImageSource::Url { .. } => "url",
        ImageSource::Base64 { media_type, .. } => media_type,
    };
    Some(format!("[image omitted: {label} — {reason}]"))
}

/// Split `content` into its text and the image parts that may go on the
/// wire (built by `part`, in order). Refused images become placeholder
/// lines appended to the text (no leading newline when it's empty). Shared by every converter path
/// that sends text first and pixels after it.
pub(crate) fn split_images<P>(
    content: &[ContentBlock],
    policy: ImagePolicy,
    mut part: impl FnMut(&ImageSource) -> P,
) -> (String, Vec<P>) {
    let mut text = ContentBlock::collect_text(content);
    let mut images = Vec::new();
    for block in content {
        if let ContentBlock::Image { source } = block {
            match omitted_placeholder(source, policy) {
                None => images.push(part(source)),
                Some(line) => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&line);
                }
            }
        }
    }
    (text, images)
}

/// Content parts for the follow-up user message that carries
/// tool-returned images on OpenAI-family wires, whose tool messages are
/// text-only: a caption naming the tool calls (built by `caption`), then
/// every image in call order. `None` when no call returned an image.
pub(crate) fn tool_images_message_parts<P>(
    deferred: Vec<(String, Vec<P>)>,
    caption: impl FnOnce(String) -> P,
) -> Option<Vec<P>> {
    if deferred.is_empty() {
        return None;
    }
    let ids: Vec<&str> = deferred.iter().map(|(id, _)| id.as_str()).collect();
    let text = format!("[image(s) returned by tool call(s) {}]", ids.join(", "));
    let mut parts = vec![caption(text)];
    parts.extend(deferred.into_iter().flat_map(|(_, images)| images));
    Some(parts)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn sendable(source: &ImageSource, policy: ImagePolicy) -> bool {
        omitted_placeholder(source, policy).is_none()
    }

    fn image_placeholder_line(source: &ImageSource, policy: ImagePolicy) -> String {
        omitted_placeholder(source, policy).expect("refused")
    }

    /// `len` base64 chars that start with `media_type`'s signature.
    fn b64(media_type: &str, len: usize) -> ImageSource {
        let magic = match media_type {
            "image/png" => "iVBORw0KGgo",
            "image/jpeg" => "/9j/",
            "image/gif" => "R0lGOD",
            "image/webp" => "UklGRiQAAABXRUJQ",
            _ => "",
        };
        ImageSource::Base64 {
            media_type: media_type.into(),
            data: format!("{magic}{}", "A".repeat(len - magic.len())),
        }
    }

    #[test]
    fn whitelisted_small_images_are_sendable() {
        // Without this: an over-strict gate silently turns vision off — say a
        // type added to INLINE_MEDIA_TYPES without a sniffer signature, which
        // would be refused as mislabelled on every send.
        for t in ImageSource::INLINE_MEDIA_TYPES {
            assert!(sendable(&b64(t, 100), ImagePolicy::anthropic(true)), "{t}");
        }
        let url = ImageSource::Url {
            url: "https://example.com/a.bmp".into(),
        };
        assert!(sendable(&url, ImagePolicy::anthropic(true)));
    }

    #[test]
    fn nothing_is_sendable_without_vision() {
        // Without this: a text-only model is sent image parts and the request
        // 400s — URL images too, which skip every other local check.
        let png = b64("image/png", 100);
        let url = ImageSource::Url {
            url: "https://example.com/a.png".into(),
        };
        for policy in [ImagePolicy::anthropic(false), ImagePolicy::openai(false)] {
            assert_eq!(
                image_placeholder_line(&png, policy),
                "[image omitted: image/png — current model has no vision]"
            );
            assert_eq!(
                image_placeholder_line(&url, policy),
                "[image omitted: url — current model has no vision]"
            );
        }
    }

    /// A 1×1 GIF with `frames` image descriptors, base64-encoded.
    pub(crate) fn gif_base64(frames: usize) -> String {
        let mut gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00".to_vec();
        gif.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00]); // palette
        for _ in 0..frames {
            // Graphic control extension (a delay, as animations carry),
            // then a 1×1 image descriptor with a minimal LZW body.
            gif.extend_from_slice(&[0x21, 0xF9, 0x04, 0x00, 0x0A, 0x00, 0x00, 0x00]);
            gif.extend_from_slice(&[0x2C, 0, 0, 0, 0, 1, 0, 1, 0, 0x00]);
            gif.extend_from_slice(&[0x02, 0x02, 0x44, 0x01, 0x00]);
        }
        gif.push(0x3B);
        BASE64.encode(&gif)
    }

    #[test]
    fn animated_gif_is_refused_only_where_static_gifs_are_required() {
        // Without this: an animated GIF already in history (written before
        // Read reduced animations to one frame) or returned by a custom tool
        // reaches an OpenAI-family wire, which accepts only non-animated
        // GIFs, and every later request in the thread is rejected.
        let animated = ImageSource::Base64 {
            media_type: "image/gif".into(),
            data: gif_base64(2),
        };
        assert!(!sendable(&animated, ImagePolicy::openai(true)));
        assert!(
            image_placeholder_line(&animated, ImagePolicy::openai(true)).contains("animated GIF")
        );
        // Anthropic accepts animated GIFs — dropping it there would lose
        // the image for nothing.
        assert!(sendable(&animated, ImagePolicy::anthropic(true)));
        // Static GIFs, and data that doesn't parse as a GIF, still pass.
        let still = ImageSource::Base64 {
            media_type: "image/gif".into(),
            data: gif_base64(1),
        };
        assert!(sendable(&still, ImagePolicy::openai(true)));
        // A second frame cut off mid-data still makes it animated (Read
        // agrees, so the two never disagree on the same file).
        let mut bytes = BASE64.decode(gif_base64(2).as_bytes()).unwrap();
        bytes.truncate(bytes.len() - 3);
        let truncated = ImageSource::Base64 {
            media_type: "image/gif".into(),
            data: BASE64.encode(&bytes),
        };
        assert!(!sendable(&truncated, ImagePolicy::openai(true)));
        let garbled = ImageSource::Base64 {
            media_type: "image/gif".into(),
            data: "R0lGODlhAQABAIAAAP".into(),
        };
        assert!(sendable(&garbled, ImagePolicy::openai(true)));
    }

    #[test]
    fn mislabelled_payload_degrades_with_reason() {
        // Without this: a JPEG labelled image/png (history written before
        // Read sniffed content) passes the gate, Anthropic rejects the
        // mismatch, and every later request in the thread fails the same way.
        let jpeg_as_png = ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "/9j/4AAQSkZJRgABAQ".into(),
        };
        assert!(!sendable(&jpeg_as_png, ImagePolicy::anthropic(true)));
        assert!(
            image_placeholder_line(&jpeg_as_png, ImagePolicy::anthropic(true))
                .contains("does not match")
        );
        // Each signature is recognised: correctly labelled it passes, under
        // any other whitelisted label it is refused.
        let b64 = |media_type: &str, data: &str| ImageSource::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        };
        let samples = [
            ("image/jpeg", "/9j/4AAQSkZJRgABAQ"),
            ("image/png", "iVBORw0KGgoAAAANSUhEUg"),
            ("image/gif", "R0lGODlhAQABAIAAAP"),
            ("image/webp", "UklGRiQAAABXRUJQVlA4IBgAAA"),
        ];
        for (actual, data) in samples {
            for label in ImageSource::INLINE_MEDIA_TYPES {
                assert_eq!(
                    sendable(&b64(label, data), ImagePolicy::anthropic(true)),
                    label == actual,
                    "{label} {data}"
                );
            }
        }
    }

    #[test]
    fn unrecognisable_payload_degrades_with_reason() {
        // Without this: an "image" that is no image at all — an HTML error
        // page saved as `.png` and labelled by extension in history written
        // before Read sniffed content — reaches the provider, which rejects
        // it on every later request in the thread.
        let html_as_png = ImageSource::Base64 {
            media_type: "image/png".into(),
            data: "PCFET0NUWVBFIGh0bWw+".into(), // "<!DOCTYPE html>"
        };
        assert!(!sendable(&html_as_png, ImagePolicy::anthropic(true)));
        assert!(
            image_placeholder_line(&html_as_png, ImagePolicy::anthropic(true))
                .contains("does not match")
        );
    }

    #[test]
    fn oversized_payload_degrades_with_reason() {
        // Without this: an image over the 5 MB per-image cap 400s the
        // request on every turn it is replayed.
        let src = b64("image/png", ImageSource::MAX_INLINE_BASE64_BYTES + 1);
        assert!(!sendable(&src, ImagePolicy::anthropic(true)));
        assert!(image_placeholder_line(&src, ImagePolicy::anthropic(true))
            .contains("exceeds the provider size limit"));
        assert!(sendable(
            &b64("image/png", ImageSource::MAX_INLINE_BASE64_BYTES),
            ImagePolicy::anthropic(true)
        ));
    }
}

//! Normalise images before they enter conversation history.
//!
//! A captured image is stored in SQLite and re-sent verbatim on every
//! later request in the thread, so what Read returns has to be both
//! *small* and *provider-acceptable*:
//!
//! - **Format by content, not extension.** The real format is sniffed
//!   from magic bytes; a `.png` that is actually a JPEG gets the right
//!   media type, and a non-image gets a clean error.
//! - **Upright.** EXIF orientation is applied, so a portrait phone photo
//!   (stored sideways with a rotate tag) isn't shown to the model on its
//!   side once re-encoding drops the tag.
//! - **Validated.** Every image is decoded (under allocation and dimension
//!   limits, so a decompression bomb fails fast; GIFs to their first
//!   frame). A file that won't decode never reaches the provider, where it
//!   would 400 the request.
//! - **Static.** An animated GIF is reduced to its first frame: OpenAI
//!   only accepts non-animated GIFs, and a refused image would be replayed
//!   from history on every later request.
//! - **Passthrough when already fine.** PNG / JPEG / static GIF / WebP
//!   within the [`ImageLimits`] byte and edge budget keep their original
//!   bytes (transparency intact). The byte budget is measured on the
//!   base64 payload — what history and the wire actually carry.
//! - **Otherwise re-encoded.** Downscaled to fit the edge cap, alpha
//!   flattened onto white, JPEG at descending quality — shrinking further
//!   if needed — until the result fits the byte budget. This is also how
//!   BMP / TIFF become something every provider takes.
//!
//! The budget comes from `ToolContext::image_limits` (operator config,
//! `[engine] image_max_bytes` / `image_max_edge_px`), clamped here so a
//! hand-built SDK context can't push it out of range.

use data_encoding::BASE64;
use image::codecs::jpeg::JpegEncoder;
use image::metadata::Orientation;
use image::{
    DynamicImage, ImageDecoder, ImageError, ImageFormat, ImageReader, Limits, Rgb, RgbImage,
};
use snaca_core::ImageSource;
use snaca_tools_api::ImageLimits;
use std::io::Cursor;
use std::num::NonZeroU64;

const JPEG_QUALITIES: [u8; 4] = [85, 70, 55, 40];

/// Decode guards against decompression bombs.
const MAX_DECODE_EDGE: u32 = 16_384;
const MAX_DECODE_ALLOC: u64 = 256 * 1024 * 1024;

/// An image ready to embed.
#[derive(Debug)]
pub(crate) struct PreparedImage {
    pub media_type: &'static str,
    pub bytes: Vec<u8>,
    /// Set when the pixels were re-encoded: tells the model what it is
    /// looking at (scaled coordinates, lost transparency, first frame).
    pub note: Option<String>,
}

/// Validate and normalise `bytes` for embedding within `budget`. `Err`
/// carries a human-readable reason (unrecognised format, corrupt data, …).
pub(crate) fn prepare_image(bytes: Vec<u8>, budget: ImageLimits) -> Result<PreparedImage, String> {
    let budget = budget.clamped();
    let mut reader = ImageReader::new(Cursor::new(bytes.as_slice()))
        .with_guessed_format()
        .map_err(|e| format!("cannot read image: {e}"))?;
    let format = reader
        .format()
        .ok_or_else(|| "not a recognised image format".to_string())?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DECODE_EDGE);
    limits.max_image_height = Some(MAX_DECODE_EDGE);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits.clone());
    let decode_err = |e: ImageError| match e {
        // Say what happened and what to do: the raw limits error reads
        // like a corrupt file, and the model would just retry the Read.
        ImageError::Limits(_) => format!(
            "image too large to process: decoding this {} exceeds the {} MiB / {} px \
             safety limits; send a smaller version",
            format_label(format),
            MAX_DECODE_ALLOC / (1024 * 1024),
            MAX_DECODE_EDGE
        ),
        e => format!("cannot decode {} image: {e}", format_label(format)),
    };
    let mut decoder = reader.into_decoder().map_err(decode_err)?;
    // `into_decoder` checks only dimensions; the allocation cap is checked
    // by `ImageReader::decode`, which can't hand back the EXIF orientation.
    // Do its check here, before `from_decoder` allocates the output buffer
    // — a 16384² image passes the edge limit yet needs ~1 GiB — and hand
    // the decoder only what's left, as `decode` does, so its own buffers
    // (a GIF frame offset from the screen gets one) share the same cap.
    limits.reserve(decoder.total_bytes()).map_err(decode_err)?;
    decoder.set_limits(limits).map_err(decode_err)?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(decode_err)?;
    img.apply_orientation(orientation);

    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        // A 0×0 GIF decodes without error, and would pass through below
        // as a valid image that providers reject.
        return Err(format!("{} image has no pixels", format_label(format)));
    }
    // A rotate/flip tag rules out passthrough: the original bytes are
    // still stored sideways, and not every provider honours EXIF.
    let upright = orientation == Orientation::NoTransforms;
    // `img` already holds only the first frame; the original bytes would
    // carry the animation to the wire.
    let animated = format == ImageFormat::Gif && is_animated_gif(&bytes);
    if let Some(media_type) = passthrough_media_type(format) {
        if upright
            && !animated
            && BASE64.encode_len(bytes.len()) <= budget.max_bytes
            && w.max(h) <= budget.max_edge_px
        {
            return Ok(PreparedImage {
                media_type,
                bytes,
                note: None,
            });
        }
    }

    let (out, (ow, oh)) = shrink_to_budget(&img, budget)?;
    Ok(PreparedImage {
        media_type: "image/jpeg",
        note: Some(format!(
            "[image re-encoded for the model: {} {w}x{h}{}, {} bytes -> JPEG {ow}x{oh}, {} bytes]",
            format_label(format),
            if animated {
                " (animated; first frame only)"
            } else {
                ""
            },
            bytes.len(),
            out.len()
        )),
        bytes: out,
    })
}

/// More than one frame. Counts frame descriptors with LZW decoding
/// skipped, so no frame's pixel buffer is ever allocated: a later frame's
/// descriptor is never validated by the first-frame decode, and one that
/// claims 65535×65535 would otherwise be a ~17 GB allocation — an abort,
/// not an error. The memory cap is a second line of defence.
fn is_animated_gif(bytes: &[u8]) -> bool {
    let mut options = gif::DecodeOptions::new();
    options.skip_frame_decoding(true);
    // Skipped like any extension, as the send-time gate's block walker does.
    options.allow_unknown_blocks(true);
    options.set_memory_limit(gif::MemoryLimit::Bytes(
        NonZeroU64::new(MAX_DECODE_ALLOC).expect("non-zero"),
    ));
    let Ok(mut decoder) = options.read_info(bytes) else {
        return false;
    };
    // Count frame descriptors: `next_frame_info` skips each frame's data,
    // so a second frame is detected however broken its data is. A file cut
    // off after its only frame (a missing trailer) stays static; any other
    // error past the first frame — say a second frame with no colour table,
    // which the gate on the wire still counts — reads as animated, since
    // re-encoding the first frame is always safe.
    let mut frames = 0;
    loop {
        match decoder.next_frame_info() {
            Ok(Some(_)) => {
                frames += 1;
                if frames > 1 {
                    return true;
                }
            }
            Ok(None) | Err(gif::DecodingError::UnexpectedEof | gif::DecodingError::Io(_)) => {
                return false
            }
            Err(_) => return frames >= 1,
        }
    }
}

/// Media type for `format` if every vision provider accepts it inline —
/// the same whitelist the `snaca-llm` send-time gate enforces.
fn passthrough_media_type(format: ImageFormat) -> Option<&'static str> {
    let media_type = format.to_mime_type();
    ImageSource::INLINE_MEDIA_TYPES
        .contains(&media_type)
        .then_some(media_type)
}

fn format_label(format: ImageFormat) -> &'static str {
    format
        .extensions_str()
        .first()
        .copied()
        .unwrap_or("unknown")
}

/// Downscale + JPEG-encode until the result's base64 fits `budget.max_bytes`,
/// shrinking no further than `ImageLimits::MIN_EDGE_PX` (past that the
/// image stops being useful to look at). Returns the encoded bytes and
/// their dimensions.
fn shrink_to_budget(
    img: &DynamicImage,
    budget: ImageLimits,
) -> Result<(Vec<u8>, (u32, u32)), String> {
    // Scale with area-averaging `thumbnail`: one integer pass, no float
    // intermediate the size of the source (`resize` needs 16 bytes a pixel
    // — 768 MB from an 8000 px base). The full-size decode is scaled down
    // once; every later step shrinks that copy, not the original.
    let fitted;
    let base = if img.width().max(img.height()) > budget.max_edge_px {
        fitted = img.thumbnail(budget.max_edge_px, budget.max_edge_px);
        &fitted
    } else {
        img
    };
    let longest = base.width().max(base.height());
    let mut edge = longest;
    loop {
        let scaled;
        let source = if longest > edge {
            scaled = base.thumbnail(edge, edge);
            &scaled
        } else {
            base
        };
        let rgb = flatten_on_white(source);
        for quality in JPEG_QUALITIES {
            let mut out = Vec::new();
            JpegEncoder::new_with_quality(&mut out, quality)
                .encode_image(&rgb)
                .map_err(|e| format!("cannot encode JPEG: {e}"))?;
            if BASE64.encode_len(out.len()) <= budget.max_bytes {
                return Ok((out, rgb.dimensions()));
            }
        }
        if edge <= ImageLimits::MIN_EDGE_PX {
            return Err(format!(
                "cannot compress image under {} bytes (base64)",
                budget.max_bytes
            ));
        }
        edge = (edge * 3 / 4).max(ImageLimits::MIN_EDGE_PX);
    }
}

/// JPEG has no alpha. Dropping the channel outright would expose whatever
/// RGB sits under transparent pixels (often black), so composite onto
/// white instead.
fn flatten_on_white(img: &DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y).0;
        let a = u16::from(p[3]);
        let blend = |c: u8| ((u16::from(c) * a + 255 * (255 - a)) / 255) as u8;
        Rgb([blend(p[0]), blend(p[1]), blend(p[2])])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};

    fn encode(img: &DynamicImage, format: ImageFormat) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        img.write_to(&mut out, format).unwrap();
        out.into_inner()
    }

    /// Per-pixel pseudo-random noise: incompressible, so byte budgets
    /// are actually exercised.
    fn noise(w: u32, h: u32) -> DynamicImage {
        let mut seed: u32 = 0x9e37_79b9;
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |_, _| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let [r, g, b, _] = seed.to_le_bytes();
            Rgb([r, g, b])
        }))
    }

    #[test]
    fn small_png_passes_through_byte_identical() {
        // Without this: every screenshot is recompressed to lossy JPEG
        // (blurring text, dropping transparency) even when it already fits.
        let bytes = encode(&noise(32, 32), ImageFormat::Png);
        let out = prepare_image(bytes.clone(), ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/png");
        assert_eq!(out.bytes, bytes);
        assert!(out.note.is_none());
    }

    #[test]
    fn bmp_is_converted_to_jpeg() {
        // Without this: a small BMP passes through as image/bmp, which no
        // provider accepts — the gate degrades it to a placeholder and the
        // model never sees it.
        let bytes = encode(&noise(40, 30), ImageFormat::Bmp);
        let out = prepare_image(bytes.clone(), ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
        assert_eq!(image::guess_format(&out.bytes).unwrap(), ImageFormat::Jpeg);
        let note = out.note.unwrap();
        assert!(note.contains("bmp 40x30"), "{note}");
    }

    #[test]
    fn oversized_bytes_shrink_to_fit_budget() {
        // Without this: a multi-MB capture enters history verbatim and is
        // re-sent on every later request until the byte cap strips it.
        let bytes = encode(&noise(3000, 2000), ImageFormat::Png);
        let budget = ImageLimits::default();
        assert!(BASE64.encode_len(bytes.len()) > budget.max_bytes);
        let out = prepare_image(bytes, budget).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
        let wire = BASE64.encode_len(out.bytes.len());
        assert!(wire <= budget.max_bytes, "{wire}");
        // Aspect ratio survives the downscale.
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        let ratio = decoded.width() as f64 / decoded.height() as f64;
        assert!((ratio - 1.5).abs() < 0.02, "{ratio}");
    }

    #[test]
    fn oversized_dimensions_are_capped_even_within_byte_budget() {
        // Without this: a flat 3000 px screenshot is a few KB, so only the
        // edge cap stops it going out at full resolution — pixels the
        // provider downsamples anyway (and >8000 px Anthropic rejects).
        let bytes = encode(
            &DynamicImage::ImageRgb8(RgbImage::from_pixel(3000, 2000, Rgb([30, 90, 200]))),
            ImageFormat::Png,
        );
        let budget = ImageLimits::default();
        assert!(BASE64.encode_len(bytes.len()) <= budget.max_bytes);
        let out = prepare_image(bytes, budget).unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (1568, 1045));
    }

    #[test]
    fn transparency_is_flattened_onto_white() {
        // Without this: JPEG drops alpha and transparent areas (icons,
        // diagrams) come out black.
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2000, 10, Rgba([0, 0, 0, 0])));
        let out = prepare_image(encode(&img, ImageFormat::Png), ImageLimits::default()).unwrap();
        let px = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let p = px.get_pixel(0, 0).0;
        assert!(p.iter().all(|&c| c > 240), "expected white, got {p:?}");
    }

    /// Splice a minimal EXIF APP1 segment carrying only an Orientation
    /// tag right after the JPEG SOI marker.
    fn with_exif_orientation(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        let mut app1 = b"Exif\0\0II\x2a\0\x08\0\0\0".to_vec(); // LE TIFF, IFD0 at 8
        app1.extend_from_slice(&1u16.to_le_bytes()); // one entry
        app1.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        app1.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        app1.extend_from_slice(&1u32.to_le_bytes()); // count
        app1.extend_from_slice(&orientation.to_le_bytes());
        app1.extend_from_slice(&[0, 0]); // value padding
        app1.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        out.extend(app1);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn exif_orientation_is_applied() {
        // Without this: a portrait phone photo is stored sideways with an
        // EXIF rotate tag; re-encoding drops the tag without rotating the
        // pixels, so the model sees it on its side.
        let img = RgbImage::from_fn(40, 20, |x, _| {
            if x < 20 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        });
        let jpeg = encode(&DynamicImage::ImageRgb8(img), ImageFormat::Jpeg);
        // 6 = rotate 90° clockwise: the left (red) half ends up on top.
        let tagged = with_exif_orientation(&jpeg, 6);
        let out = prepare_image(tagged, ImageLimits::default()).unwrap();
        assert!(out.note.is_some(), "a rotated image can't pass through");
        let px = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        assert_eq!(px.dimensions(), (20, 40));
        let [r, _, b] = px.get_pixel(10, 5).0;
        assert!(r > 200 && b < 60, "top should be red: {:?}", (r, b));
        let [r, _, b] = px.get_pixel(10, 35).0;
        assert!(b > 200 && r < 60, "bottom should be blue: {:?}", (r, b));
    }

    fn gif(frames: &[[u8; 3]]) -> Vec<u8> {
        use image::codecs::gif::GifEncoder;
        use image::Frame;
        let mut out = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut out);
            encoder
                .encode_frames(frames.iter().map(|&[r, g, b]| {
                    Frame::new(RgbaImage::from_pixel(32, 32, Rgba([r, g, b, 255])))
                }))
                .unwrap();
        }
        out
    }

    #[test]
    fn animated_gif_is_reduced_to_its_first_frame() {
        // Without this: a small animated GIF passes through with all its
        // frames, and OpenAI only accepts non-animated GIFs.
        let bytes = gif(&[[255, 0, 0], [0, 0, 255]]);
        let out = prepare_image(bytes, ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
        let note = out.note.unwrap();
        assert!(note.contains("animated; first frame only"), "{note}");
        let px = image::load_from_memory(&out.bytes).unwrap().to_rgb8();
        let [r, _, b] = px.get_pixel(16, 16).0;
        assert!(r > 200 && b < 60, "first (red) frame kept: {:?}", (r, b));
    }

    #[test]
    fn truncated_static_gif_is_not_taken_for_animated() {
        // Without this: a static GIF missing its trailer byte (common in
        // files cut short by a chat upload) errors after its first frame,
        // was judged animated, and got needlessly re-encoded as lossy JPEG.
        let mut bytes = gif(&[[0, 128, 0]]);
        assert_eq!(bytes.pop(), Some(0x3B), "fixture ends in the trailer");
        let out = prepare_image(bytes.clone(), ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/gif");
        assert_eq!(out.bytes, bytes);
    }

    #[test]
    fn animated_gif_with_a_truncated_second_frame_is_still_animated() {
        // Without this: a GIF whose second frame is cut off mid-data would
        // pass through as static, and OpenAI-family wires (whose send-time
        // gate counts the frame) would refuse it on every request.
        let mut bytes = gif(&[[255, 0, 0], [0, 0, 255]]);
        bytes.truncate(bytes.len() - 4);
        let out = prepare_image(bytes, ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
    }

    #[test]
    fn oversized_dimensions_fail_with_an_actionable_message() {
        // Without this: a photo past the decode safety limits fails with a
        // raw limits error that reads like a corrupt file, so the model
        // retries the Read instead of asking for a smaller image.
        let mut bmp = encode(&noise(1, 1), ImageFormat::Bmp);
        bmp[18..22].copy_from_slice(&20_000i32.to_le_bytes()); // width
        bmp[22..26].copy_from_slice(&20_000i32.to_le_bytes()); // height
        let err = prepare_image(bmp, ImageLimits::default()).unwrap_err();
        assert!(err.contains("too large to process"), "{err}");
    }

    #[test]
    fn allocation_cap_is_enforced_within_the_edge_limit() {
        // Without this: 16000² passes the 16384 px edge check, so decoding
        // would allocate ~768 MB per Read — several concurrent Reads of
        // such a file (a few KB on the wire) take the whole server down.
        let mut bmp = encode(&noise(1, 1), ImageFormat::Bmp);
        bmp[18..22].copy_from_slice(&16_000i32.to_le_bytes()); // width
        bmp[22..26].copy_from_slice(&16_000i32.to_le_bytes()); // height
        let err = prepare_image(bmp, ImageLimits::default()).unwrap_err();
        assert!(err.contains("too large to process"), "{err}");
    }

    /// A GIF with an `sw`×`sh` logical screen (no global palette) and one
    /// frame per `(left, w, h, palette)`: a local 2-colour table when
    /// `palette`, then a minimal LZW body.
    fn gif_with_frames(sw: u16, sh: u16, frames: &[(u16, u16, u16, bool)]) -> Vec<u8> {
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&sw.to_le_bytes());
        gif.extend_from_slice(&sh.to_le_bytes());
        gif.extend_from_slice(&[0x00, 0x00, 0x00]);
        for &(left, w, h, palette) in frames {
            gif.push(0x2C);
            gif.extend_from_slice(&left.to_le_bytes());
            gif.extend_from_slice(&[0, 0]);
            gif.extend_from_slice(&w.to_le_bytes());
            gif.extend_from_slice(&h.to_le_bytes());
            if palette {
                gif.push(0x80);
                gif.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00]);
            } else {
                gif.push(0x00);
            }
            gif.extend_from_slice(&[0x02, 0x02, 0x44, 0x01, 0x00]);
        }
        gif.push(0x3B);
        gif
    }

    #[test]
    fn decoder_buffers_share_the_allocation_cap() {
        // Without this: the cap is checked against the output buffer only,
        // and the decoder keeps its own full 256 MiB for internal buffers.
        // An 8000² GIF whose first frame is offset from the screen edge
        // needs a second ~256 MB frame buffer — two concurrent Reads of
        // such a file (a few dozen bytes) would hold ~1 GiB.
        let bytes = gif_with_frames(8000, 8000, &[(1, 7999, 8000, true)]);
        let err = prepare_image(bytes, ImageLimits::default()).unwrap_err();
        assert!(err.contains("too large to process"), "{err}");
    }

    #[test]
    fn zero_sized_image_is_rejected() {
        // Without this: a 0×0 GIF decodes cleanly and passes through as a
        // "valid" image, which the provider then rejects on every replay.
        let bytes = gif_with_frames(0, 0, &[(0, 1, 1, true)]);
        let err = prepare_image(bytes, ImageLimits::default()).unwrap_err();
        assert!(err.contains("no pixels"), "{err}");
    }

    #[test]
    fn second_frame_the_decoder_rejects_still_counts_as_animated() {
        // Without this: a second frame with no colour table stops the gif
        // crate, so Read saw one frame and passed the file through as a
        // static GIF — while the send-time gate counts two and replaces it
        // with a placeholder on OpenAI-family wires.
        let bytes = gif_with_frames(1, 1, &[(0, 1, 1, true), (0, 1, 1, false)]);
        let out = prepare_image(bytes, ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
    }

    #[test]
    fn unknown_extension_does_not_make_a_static_gif_animated() {
        // Without this: the gif crate stops at an extension label it
        // doesn't know, which past the first frame read as "animated", so
        // a static GIF carrying one was needlessly re-encoded as JPEG (the
        // send-time gate skips any extension the same way).
        let mut bytes = gif_with_frames(1, 1, &[(0, 1, 1, true)]);
        let trailer = bytes.pop();
        bytes.extend_from_slice(&[0x21, 0x99, 0x01, 0x00, 0x00]);
        bytes.extend(trailer);
        let out = prepare_image(bytes.clone(), ImageLimits::default()).unwrap();
        assert_eq!(out.media_type, "image/gif");
        assert_eq!(out.bytes, bytes);
    }

    #[test]
    fn corrupt_or_unknown_bytes_are_rejected() {
        // Without this: a truncated file is stored as an image block, the
        // provider 400s on it, and every later request replays the error.
        let mut truncated = encode(&noise(64, 64), ImageFormat::Png);
        truncated.truncate(truncated.len() / 2);
        assert!(prepare_image(truncated, ImageLimits::default()).is_err());
        assert!(
            prepare_image(b"definitely not an image".to_vec(), ImageLimits::default()).is_err()
        );
    }

    #[test]
    fn budget_is_measured_on_base64_not_raw_bytes() {
        // Without this: an image just under the budget in raw bytes passes
        // through though its base64 — what history stores and the wire
        // carries — is a third larger, overshooting the cap.
        // ~216 KB raw is under 256 KiB, but its ~288 KB base64 is not.
        let bytes = encode(&noise(300, 240), ImageFormat::Png);
        let budget = ImageLimits::default();
        assert!(bytes.len() <= budget.max_bytes);
        assert!(BASE64.encode_len(bytes.len()) > budget.max_bytes);
        let out = prepare_image(bytes.clone(), budget).unwrap();
        assert_eq!(out.media_type, "image/jpeg");
        assert!(BASE64.encode_len(out.bytes.len()) <= budget.max_bytes);
    }

    #[test]
    fn configured_budget_is_honoured() {
        // Without this: `[engine] image_max_bytes` / `image_max_edge_px`
        // would be accepted by config and silently ignored here.
        // ~300 KB PNG at 600 px: over the default byte budget, so it is
        // re-encoded by default…
        let bytes = encode(&noise(600, 160), ImageFormat::Png);
        assert!(BASE64.encode_len(bytes.len()) > ImageLimits::DEFAULT_MAX_BYTES);
        assert!(prepare_image(bytes.clone(), ImageLimits::default())
            .unwrap()
            .note
            .is_some());
        // …but passes through untouched under a 1 MB budget.
        let roomy = ImageLimits {
            max_bytes: 1024 * 1024,
            ..ImageLimits::default()
        };
        let out = prepare_image(bytes.clone(), roomy).unwrap();
        assert_eq!(out.bytes, bytes);

        // A tighter edge cap downscales even a small image.
        let small_edge = ImageLimits {
            max_edge_px: 300,
            ..ImageLimits::default()
        };
        let out = prepare_image(bytes.clone(), small_edge).unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!(decoded.width(), 300);
    }

    #[test]
    fn out_of_range_budget_is_clamped() {
        // Without this: an SDK-built context with a 0-byte budget or a 1 px
        // edge makes every image Read fail ("cannot compress") or shrink to
        // a dot.
        let bytes = encode(&noise(1000, 1000), ImageFormat::Png);
        let absurd = ImageLimits {
            max_bytes: 0,
            max_edge_px: 1,
        };
        let out = prepare_image(bytes.clone(), absurd).unwrap();
        let decoded = image::load_from_memory(&out.bytes).unwrap();
        assert!(decoded.width() >= ImageLimits::MIN_EDGE_PX);
        assert!(BASE64.encode_len(out.bytes.len()) <= ImageLimits::MIN_BYTES);
    }
}

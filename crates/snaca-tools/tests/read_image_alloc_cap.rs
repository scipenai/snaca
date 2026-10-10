//! Read on a GIF frame bomb, under an allocator that refuses any single
//! allocation over `image_prep`'s 256 MiB decode cap.
//!
//! Zeroed allocations are lazy: on macOS, and on Linux with default
//! overcommit, reserving a 17 GB buffer that is never written succeeds. A
//! regression that sizes a frame buffer from a hostile descriptor would
//! pass an ordinary test and abort the server only on hosts with strict
//! overcommit or an address-space limit. Here any such allocation returns
//! null and the process aborts, on every host. Its own binary, because the
//! allocator is process-wide.

use serde_json::json;
use snaca_core::{ContentBlock, ImageSource, ProjectId, SessionId, TenantId};
use snaca_tools::read::ReadTool;
use snaca_tools_api::{Tool, ToolContext, ToolOutput};
use std::alloc::{GlobalAlloc, Layout, System};

const MAX_SINGLE_ALLOC: usize = 256 * 1024 * 1024;

struct Capped;

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > MAX_SINGLE_ALLOC {
            return std::ptr::null_mut();
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > MAX_SINGLE_ALLOC {
            return std::ptr::null_mut();
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > MAX_SINGLE_ALLOC {
            return std::ptr::null_mut();
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: Capped = Capped;

/// A 1×1 GIF whose *second* frame descriptor claims 65535×65535 —
/// about 50 bytes on the wire.
fn gif_with_huge_second_frame() -> Vec<u8> {
    let mut gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00".to_vec();
    gif.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00]); // palette
    let frame = |w: u16, h: u16| {
        let mut f = vec![0x2C, 0, 0, 0, 0];
        f.extend_from_slice(&w.to_le_bytes());
        f.extend_from_slice(&h.to_le_bytes());
        f.extend_from_slice(&[0x00, 0x02, 0x02, 0x44, 0x01, 0x00]);
        f
    };
    gif.extend(frame(1, 1));
    gif.extend(frame(u16::MAX, u16::MAX));
    gif.push(0x3B);
    gif
}

#[tokio::test]
async fn gif_frame_bomb_is_counted_without_decoding_it() {
    // Without this: an animation check that decodes frame 2 sizes its buffer
    // from the descriptor — a ~17 GB allocation that aborts the process
    // (every tenant) where overcommit is strict — and one that shrugs off
    // the absurd descriptor passes the GIF through as static, which
    // OpenAI-family wires (whose gate counts both frames) then refuse.
    assert!(
        Vec::<u8>::new()
            .try_reserve_exact(MAX_SINGLE_ALLOC + 1)
            .is_err(),
        "the allocation cap is armed"
    );
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bomb.gif"), gif_with_huge_second_frame()).unwrap();
    let ctx = ToolContext::new(
        TenantId::new("t"),
        ProjectId::from_raw("p"),
        SessionId::new(),
        dir.path().to_path_buf(),
    );

    let out = ReadTool
        .execute(json!({"path": "bomb.gif"}), &ctx)
        .await
        .unwrap();
    let ToolOutput::Blocks(blocks) = out else {
        panic!("expected Blocks output");
    };
    assert!(
        matches!(blocks.as_slice(), [
            ContentBlock::Text { text },
            ContentBlock::Image { source: ImageSource::Base64 { media_type, .. } },
        ] if text.contains("animated") && media_type == "image/jpeg"),
        "{blocks:?}"
    );
}

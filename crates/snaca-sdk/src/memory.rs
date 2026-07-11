//! Memory helpers for SDK users.

use crate::{ensure_absolute, Result};
pub use snaca_agent_api::{
    MemoryEntryData, MemoryIndexRequest, MemoryListRequest, MemoryProvider, MemoryProviderError,
    MemoryProviderSlot, MemoryReadRequest, MemoryWriteRequest,
};
// Store side (R5/M1): the built-in file-tree store a host reads/writes directly.
use snaca_memory::FileTreeMemoryProvider;
pub use snaca_memory::{MemoryError, MemoryScope, MemoryStore};
use snaca_workspace::WorkspaceLayout;
use std::path::PathBuf;

/// Build a `FileTreeMemoryProvider` rooted at `data_root`. The
/// provider stores memory entries as plain markdown files under
/// `<data_root>/<tenant>/projects/<project>/memory/`.
pub fn file_tree(data_root: impl Into<PathBuf>) -> Result<FileTreeMemoryProvider> {
    let root = ensure_absolute(data_root.into())?;
    Ok(FileTreeMemoryProvider::new(WorkspaceLayout::new(root)?))
}

//! Memory helpers for SDK users.

use crate::{ensure_absolute, Result};
pub use snaca_agent_api::{
    MemoryEntryData, MemoryIndexRequest, MemoryListRequest, MemoryProvider, MemoryProviderError,
    MemoryProviderSlot, MemoryReadRequest, MemoryRecallHit, MemoryRecallRequest,
    MemoryWriteRequest,
};
use snaca_memory::{Embedder, FileTreeMemoryProvider};
use snaca_state::Database;
use snaca_workspace::WorkspaceLayout;
use std::path::PathBuf;
use std::sync::Arc;

pub fn file_tree(data_root: impl Into<PathBuf>) -> Result<FileTreeMemoryProvider> {
    let root = ensure_absolute(data_root.into())?;
    Ok(FileTreeMemoryProvider::new(WorkspaceLayout::new(root)?))
}

pub fn file_tree_with_index(
    data_root: impl Into<PathBuf>,
    state: Database,
    embedder: Arc<dyn Embedder>,
) -> Result<FileTreeMemoryProvider> {
    Ok(file_tree(data_root)?.with_index(state, embedder))
}

pub fn file_tree_with_hash_index(
    data_root: impl Into<PathBuf>,
    state: Database,
) -> Result<FileTreeMemoryProvider> {
    Ok(file_tree(data_root)?.with_hash_index(state))
}

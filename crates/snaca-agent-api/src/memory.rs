//! Memory provider contracts for embeddable agent runtimes.

use async_trait::async_trait;
use snaca_core::{ProjectId, TenantId};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct MemoryWriteRequest {
    pub tenant_id: TenantId,
    pub project_id: ProjectId,
    /// Provider-defined scope name. The built-in provider accepts
    /// `user`, `project`, `reference`, and `feedback`.
    pub scope: String,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct MemoryReadRequest {
    pub tenant_id: TenantId,
    pub project_id: ProjectId,
    pub scope: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct MemoryRecallRequest {
    pub tenant_id: TenantId,
    pub project_id: ProjectId,
    pub query: String,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryIndexRequest {
    pub tenant_id: TenantId,
    pub project_id: ProjectId,
}

#[derive(Debug, Clone)]
pub struct MemoryListRequest {
    pub tenant_id: TenantId,
    pub project_id: ProjectId,
    pub scope: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntryData {
    pub scope: String,
    pub name: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecallHit {
    pub scope: String,
    pub name: String,
    pub content: String,
    pub score: Option<f32>,
}

#[derive(Debug, Error)]
pub enum MemoryProviderError {
    #[error("invalid memory scope {0}")]
    InvalidScope(String),

    #[error("memory entry not found: {scope}/{name}")]
    NotFound { scope: String, name: String },

    #[error("memory provider io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("memory provider failed: {0}")]
    Other(String),
}

#[async_trait]
pub trait MemoryProvider: Send + Sync {
    async fn index(&self, request: MemoryIndexRequest) -> Result<String, MemoryProviderError>;

    async fn list(&self, request: MemoryListRequest) -> Result<Vec<String>, MemoryProviderError>;

    async fn write(
        &self,
        request: MemoryWriteRequest,
    ) -> Result<MemoryEntryData, MemoryProviderError>;

    async fn read(
        &self,
        request: MemoryReadRequest,
    ) -> Result<MemoryEntryData, MemoryProviderError>;

    async fn recall(
        &self,
        request: MemoryRecallRequest,
    ) -> Result<Vec<MemoryRecallHit>, MemoryProviderError>;
}

/// Sized wrapper used to stash an `Arc<dyn MemoryProvider>` in an opaque
/// `Arc<dyn Any>` slot. This mirrors `QuestionGateSlot` and keeps
/// `snaca-tools-api` free of higher-level runtime dependencies.
pub struct MemoryProviderSlot(pub Arc<dyn MemoryProvider>);

impl MemoryProviderSlot {
    pub fn new(provider: Arc<dyn MemoryProvider>) -> Self {
        Self(provider)
    }

    pub fn provider(&self) -> Arc<dyn MemoryProvider> {
        self.0.clone()
    }
}

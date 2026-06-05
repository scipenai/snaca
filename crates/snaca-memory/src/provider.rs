//! `snaca-agent-api::MemoryProvider` adapter for the file-tree memory store.

use crate::{HashEmbedder, IndexedMemoryStore, MemoryError, MemoryScope, MemoryStore};
use async_trait::async_trait;
use snaca_agent_api::{
    MemoryEntryData, MemoryIndexRequest, MemoryListRequest, MemoryProvider, MemoryProviderError,
    MemoryReadRequest, MemoryRecallHit, MemoryRecallRequest, MemoryWriteRequest,
};
use snaca_core::{ProjectId, TenantId};
use snaca_state::Database;
use snaca_workspace::WorkspaceLayout;
use std::sync::Arc;

#[derive(Clone)]
pub struct FileTreeMemoryProvider {
    workspace: WorkspaceLayout,
    state: Option<Database>,
    embedder: Option<Arc<dyn crate::Embedder>>,
}

impl FileTreeMemoryProvider {
    pub fn new(workspace: WorkspaceLayout) -> Self {
        Self {
            workspace,
            state: None,
            embedder: None,
        }
    }

    pub fn with_index(mut self, state: Database, embedder: Arc<dyn crate::Embedder>) -> Self {
        self.state = Some(state);
        self.embedder = Some(embedder);
        self
    }

    pub fn with_hash_index(mut self, state: Database) -> Self {
        self.state = Some(state);
        self.embedder = Some(Arc::new(HashEmbedder::default()));
        self
    }

    fn store(&self, tenant: &TenantId, project: &ProjectId) -> MemoryStore {
        MemoryStore::new(self.workspace.memory_dir(tenant, project))
    }

    fn indexed(&self, tenant: &TenantId, project: &ProjectId) -> Option<IndexedMemoryStore> {
        Some(IndexedMemoryStore::new(
            self.store(tenant, project),
            self.state.clone()?,
            self.embedder.clone()?,
            tenant.clone(),
            project.clone(),
        ))
    }
}

#[async_trait]
impl MemoryProvider for FileTreeMemoryProvider {
    async fn index(&self, request: MemoryIndexRequest) -> Result<String, MemoryProviderError> {
        self.workspace
            .ensure_project(&request.tenant_id, &request.project_id)
            .map_err(map_workspace_error)?;
        self.store(&request.tenant_id, &request.project_id)
            .index_text()
            .await
            .map_err(map_memory_error)
    }

    async fn list(&self, request: MemoryListRequest) -> Result<Vec<String>, MemoryProviderError> {
        let scope = parse_scope(&request.scope)?;
        self.workspace
            .ensure_project(&request.tenant_id, &request.project_id)
            .map_err(map_workspace_error)?;
        self.store(&request.tenant_id, &request.project_id)
            .list(scope)
            .await
            .map_err(map_memory_error)
    }

    async fn write(
        &self,
        request: MemoryWriteRequest,
    ) -> Result<MemoryEntryData, MemoryProviderError> {
        let scope = parse_scope(&request.scope)?;
        self.workspace
            .ensure_project(&request.tenant_id, &request.project_id)
            .map_err(map_workspace_error)?;
        let entry = match self.indexed(&request.tenant_id, &request.project_id) {
            Some(indexed) => indexed
                .write(scope, &request.name, &request.content)
                .await
                .map_err(map_memory_error)?,
            None => self
                .store(&request.tenant_id, &request.project_id)
                .write(scope, &request.name, &request.content)
                .await
                .map_err(map_memory_error)?,
        };
        Ok(entry.into())
    }

    async fn read(
        &self,
        request: MemoryReadRequest,
    ) -> Result<MemoryEntryData, MemoryProviderError> {
        let scope = parse_scope(&request.scope)?;
        self.workspace
            .ensure_project(&request.tenant_id, &request.project_id)
            .map_err(map_workspace_error)?;
        let entry = self
            .store(&request.tenant_id, &request.project_id)
            .read(scope, &request.name)
            .await
            .map_err(map_memory_error)?;
        Ok(entry.into())
    }

    async fn recall(
        &self,
        request: MemoryRecallRequest,
    ) -> Result<Vec<MemoryRecallHit>, MemoryProviderError> {
        self.workspace
            .ensure_project(&request.tenant_id, &request.project_id)
            .map_err(map_workspace_error)?;
        let store = self.store(&request.tenant_id, &request.project_id);
        if let Some(indexed) = self.indexed(&request.tenant_id, &request.project_id) {
            indexed.ensure_indexed().await.map_err(map_memory_error)?;
            let hits = indexed
                .search(&request.query, request.limit)
                .await
                .map_err(map_memory_error)?;
            let mut out = Vec::with_capacity(hits.len());
            for hit in hits {
                let entry = store
                    .read(hit.scope, &hit.name)
                    .await
                    .map_err(map_memory_error)?;
                out.push(MemoryRecallHit {
                    scope: entry.scope.as_str().to_string(),
                    name: entry.name,
                    content: entry.content,
                    score: Some(hit.score),
                });
            }
            return Ok(out);
        }

        let mut out = Vec::new();
        for (scope, name) in store.list_all().await.map_err(map_memory_error)? {
            if out.len() >= request.limit {
                break;
            }
            let entry = store.read(scope, &name).await.map_err(map_memory_error)?;
            out.push(MemoryRecallHit {
                scope: entry.scope.as_str().to_string(),
                name: entry.name,
                content: entry.content,
                score: None,
            });
        }
        Ok(out)
    }
}

impl From<crate::MemoryEntry> for MemoryEntryData {
    fn from(entry: crate::MemoryEntry) -> Self {
        Self {
            scope: entry.scope.as_str().to_string(),
            name: entry.name,
            content: entry.content,
        }
    }
}

fn parse_scope(scope: &str) -> Result<MemoryScope, MemoryProviderError> {
    MemoryScope::from_dir_name(scope)
        .ok_or_else(|| MemoryProviderError::InvalidScope(scope.to_string()))
}

fn map_memory_error(error: MemoryError) -> MemoryProviderError {
    match error {
        MemoryError::InvalidName { name, reason } => {
            MemoryProviderError::Other(format!("invalid memory entry name {name:?}: {reason}"))
        }
        MemoryError::NotFound { scope, name } => MemoryProviderError::NotFound {
            scope: scope.as_str().to_string(),
            name,
        },
        MemoryError::Io(e) => MemoryProviderError::Io(e),
        MemoryError::ExternalExtractorRequired { kind, filename } => MemoryProviderError::Other(
            format!("external extractor required for {kind} file {filename:?}"),
        ),
    }
}

fn map_workspace_error(error: snaca_workspace::WorkspaceError) -> MemoryProviderError {
    match error {
        snaca_workspace::WorkspaceError::Io(e) => MemoryProviderError::Io(e),
        other => MemoryProviderError::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use snaca_agent_api::{
        MemoryIndexRequest, MemoryListRequest, MemoryProvider, MemoryReadRequest,
        MemoryRecallRequest, MemoryWriteRequest,
    };

    #[tokio::test]
    async fn file_tree_memory_provider_roundtrips_and_recalls() {
        let dir = tempfile::tempdir().unwrap();
        let provider = FileTreeMemoryProvider::new(WorkspaceLayout::new(dir.path()).unwrap());
        let tenant_id = TenantId::new("tenant");
        let project_id = ProjectId::from_raw("project");

        let written = provider
            .write(MemoryWriteRequest {
                tenant_id: tenant_id.clone(),
                project_id: project_id.clone(),
                scope: "project".into(),
                name: "Conventions".into(),
                content: "Use Rust 2021.".into(),
            })
            .await
            .unwrap();
        assert_eq!(written.scope, "project");
        assert_eq!(written.name, "conventions");

        let read = provider
            .read(MemoryReadRequest {
                tenant_id: tenant_id.clone(),
                project_id: project_id.clone(),
                scope: "project".into(),
                name: "conventions".into(),
            })
            .await
            .unwrap();
        assert_eq!(read.content, "Use Rust 2021.");

        let listed = provider
            .list(MemoryListRequest {
                tenant_id: tenant_id.clone(),
                project_id: project_id.clone(),
                scope: "project".into(),
            })
            .await
            .unwrap();
        assert_eq!(listed, vec!["conventions"]);

        let index = provider
            .index(MemoryIndexRequest {
                tenant_id: tenant_id.clone(),
                project_id: project_id.clone(),
            })
            .await
            .unwrap();
        assert!(index.contains("project/conventions"));

        let hits = provider
            .recall(MemoryRecallRequest {
                tenant_id,
                project_id,
                query: "rust".into(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "conventions");
        assert_eq!(hits[0].score, None);
    }
}

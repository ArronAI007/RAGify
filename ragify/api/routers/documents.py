import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, PROJECT_ROOT, get_kb_manager, resolve_kb_path, require_membership, require_role
from ..schemas import ClearIndexRequest, DeleteDocRequest, IndexRequest, UpdateChunkRequest
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...core.vectorstores import VectorStoreManager
from ...config import get_config
from ...mcp import IndexingPipeline

router = APIRouter()

_DATASET_ROLES = ("OWNER", "ADMIN", "EDITOR", "DATASET_OPERATOR")


@router.post("/api/tenants/{tenant_id}/index")
def index_documents(
    tenant_id: str,
    body: IndexRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        pipeline = IndexingPipeline()

    payload: dict = {}
    if body.directory_path:
        payload["directory_path"] = body.directory_path
    elif body.file_paths:
        payload["file_paths"] = body.file_paths
    elif body.kb_id:
        kb_data_dir = os.path.join(PROJECT_ROOT, "data", body.kb_id)
        if os.path.isdir(kb_data_dir):
            payload["directory_path"] = kb_data_dir

    if body.clear_vectorstore is not None:
        payload["clear_vectorstore"] = body.clear_vectorstore
    elif payload.get("directory_path"):
        payload["clear_vectorstore"] = True

    result = pipeline.run(payload)
    return {"indexing_summary": result.get("indexing_summary", {})}


@router.delete("/api/tenants/{tenant_id}/index")
def clear_index(
    tenant_id: str,
    body: ClearIndexRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    vm.clear()
    return {"success": True}


@router.get("/api/tenants/{tenant_id}/stats")
def get_stats(
    tenant_id: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        cfg = get_config()
        vm = VectorStoreManager()
        store_type = cfg.get("vectorstore.type", "unknown")
        collection_name = cfg.get("vectorstore.collection_name", "")
        persist_directory = cfg.get("vectorstore.persist_directory", "")

    return {
        "store_type": store_type,
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "doc_count": vm.get_document_count(),
    }


@router.get("/api/tenants/{tenant_id}/documents")
def list_documents(
    tenant_id: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    sources = vm.get_sources()
    return {"documents": sources, "total": len(sources)}


@router.delete("/api/tenants/{tenant_id}/documents")
def delete_document(
    tenant_id: str,
    body: DeleteDocRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    source = body.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    deleted = False
    candidates = [source, os.path.join(PROJECT_ROOT, "data", body.kb_id, os.path.basename(source))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            os.remove(candidate)
            deleted = True

    removed = vm.delete_by_source(source)
    return {"success": True, "deleted": deleted, "chunks_removed": removed}


@router.get("/api/tenants/{tenant_id}/chunks")
def list_chunks(
    tenant_id: str,
    source: str,
    kb_id: str | None = None,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    if not source.strip():
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    chunks = vm.get_chunks_by_source(source)
    return {"chunks": chunks, "total": len(chunks)}


@router.put("/api/tenants/{tenant_id}/chunks")
def update_chunk(
    tenant_id: str,
    body: UpdateChunkRequest,
    membership: Membership = Depends(require_role(*_DATASET_ROLES)),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    if not body.chunk_id:
        raise HTTPException(status_code=400, detail="缺少 chunk_id 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    ok = vm.update_chunk_content(body.chunk_id, body.content)
    return {"success": ok}

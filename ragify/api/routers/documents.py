import os

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, PROJECT_ROOT, get_kb_manager, resolve_kb_path
from ..schemas import ClearIndexRequest, DeleteDocRequest, IndexRequest, UpdateChunkRequest
from ...core.kb_manager import KBManager
from ...core.vectorstores import VectorStoreManager
from ...config import get_config
from ...mcp import IndexingPipeline

router = APIRouter()


@router.post("/api/index")
def index_documents(body: IndexRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
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


@router.delete("/api/index")
def clear_index(body: ClearIndexRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    vm.clear()
    return {"success": True}


@router.get("/api/stats")
def get_stats(kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id)
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


@router.get("/api/documents")
def list_documents(kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    sources = vm.get_sources()
    return {"documents": sources, "total": len(sources)}


@router.delete("/api/documents")
def delete_document(body: DeleteDocRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    source = body.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
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


@router.get("/api/chunks")
def list_chunks(source: str, kb_id: str | None = None, manager: KBManager = Depends(get_kb_manager)) -> dict:
    if not source.strip():
        raise HTTPException(status_code=400, detail="缺少 source 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    chunks = vm.get_chunks_by_source(source)
    return {"chunks": chunks, "total": len(chunks)}


@router.put("/api/chunks")
def update_chunk(body: UpdateChunkRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    if not body.chunk_id:
        raise HTTPException(status_code=400, detail="缺少 chunk_id 参数")

    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        vm = VectorStoreManager()

    ok = vm.update_chunk_content(body.chunk_id, body.content)
    return {"success": ok}

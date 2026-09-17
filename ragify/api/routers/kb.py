from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager
from ..schemas import CreateKBRequest
from ...config import get_config
from ...core.kb_manager import KBManager
from ...core.vectorstores import VectorStoreManager

router = APIRouter()


@router.get("/api/kb")
def list_kbs(manager: KBManager = Depends(get_kb_manager)) -> dict:
    kbs = manager.list_all()
    kbs_out = []
    with KB_LOCK:
        for kb in kbs:
            doc_count = 0
            try:
                persist_dir = manager.get_persist_dir(kb.id)
                get_config().update("vectorstore.persist_directory", persist_dir)
                vm = VectorStoreManager()
                doc_count = vm.get_document_count()
            except Exception:
                pass
            kbs_out.append({
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at,
                "doc_count": doc_count,
            })
    return {"knowledge_bases": kbs_out}


@router.post("/api/kb")
def create_kb(body: CreateKBRequest, manager: KBManager = Depends(get_kb_manager)) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    try:
        kb = manager.create(name, body.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": kb.created_at,
    }


@router.delete("/api/kb/{kb_id}")
def delete_kb(kb_id: str, manager: KBManager = Depends(get_kb_manager)) -> dict:
    ok = manager.delete(kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库 '{kb_id}' 不存在")
    return {"success": True}

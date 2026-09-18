import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager, require_membership, require_role
from ..schemas import CreateKBRequest
from ...config import get_config
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...core.vectorstores import VectorStoreManager

logger = logging.getLogger("ragify.api.routers.kb")

router = APIRouter()


@router.get("/api/tenants/{tenant_id}/kb")
def list_kbs(
    tenant_id: str,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    kbs = manager.list_all(tenant_id)
    kbs_out = []
    with KB_LOCK:
        for kb in kbs:
            doc_count = 0
            try:
                persist_dir = manager.get_persist_dir(tenant_id, kb.id)
                get_config().update("vectorstore.persist_directory", persist_dir)
                vm = VectorStoreManager()
                doc_count = vm.get_document_count()
            except Exception as e:
                logger.warning("获取知识库 '%s' (%s) 的文档数失败: %s", kb.name, kb.id, e)
            kbs_out.append({
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "created_at": kb.created_at,
                "doc_count": doc_count,
            })
    return {"knowledge_bases": kbs_out}


@router.post("/api/tenants/{tenant_id}/kb")
def create_kb(
    tenant_id: str,
    body: CreateKBRequest,
    membership: Membership = Depends(require_role("OWNER", "ADMIN", "EDITOR")),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    try:
        kb = manager.create(name, body.description, tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": kb.created_at,
    }


@router.delete("/api/tenants/{tenant_id}/kb/{kb_id}")
def delete_kb(
    tenant_id: str,
    kb_id: str,
    membership: Membership = Depends(require_role("OWNER", "ADMIN", "EDITOR")),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    ok = manager.delete(kb_id, tenant_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"知识库 '{kb_id}' 不存在")
    return {"success": True}

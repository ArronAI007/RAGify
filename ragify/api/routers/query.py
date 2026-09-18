from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import KB_LOCK, get_kb_manager, resolve_kb_path, require_membership
from ..schemas import AgenticQueryRequest, QueryRequest
from ...agentic import AgenticRAG
from ...core.kb_manager import KBManager
from ...core.tenant_manager import Membership
from ...mcp import QueryPipeline

router = APIRouter()


@router.post("/api/tenants/{tenant_id}/query")
def query(
    tenant_id: str,
    body: QueryRequest,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        pipeline = QueryPipeline()

    result = pipeline.run({
        "query": body.query,
        "k": body.k,
        "score_threshold": body.score_threshold,
    })

    retrieved = result.get("retrieved_documents", [])
    docs_out = []
    for doc in retrieved:
        docs_out.append({
            "page_content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "retrieval_score": doc.metadata.get("retrieval_score", 0),
            },
        })

    return {
        "response": result.get("response", ""),
        "response_generated": result.get("response_generated", False),
        "retrieved_documents": docs_out,
        "query_summary": result.get("query_summary", {}),
    }


@router.post("/api/tenants/{tenant_id}/query/agentic")
def agentic_query(
    tenant_id: str,
    body: AgenticQueryRequest,
    membership: Membership = Depends(require_membership),
    manager: KBManager = Depends(get_kb_manager),
) -> dict:
    # 整个 AgenticRAG 构造 + .run() 都在锁内——它的 retrieve_docs 工具在
    # run() 执行期间（不是构造时）才现读一次全局 vectorstore 配置，所以
    # 不能像 query() 那样提前把锁放掉。见 dependencies.py 顶部的并发说明。
    with KB_LOCK:
        try:
            resolve_kb_path(manager, body.kb_id, tenant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        agent = AgenticRAG(kb_id=body.kb_id, max_iterations=body.max_iterations)
        result = agent.run(body.query, chat_history=body.chat_history)
    return result

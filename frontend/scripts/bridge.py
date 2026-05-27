#!/usr/bin/env python3
"""RAGify bridge — JSON-in/JSON-out adapter for the Next.js API layer.

Reads a JSON command from stdin, runs the matching RAGify pipeline,
and writes a JSON response to stdout.

Commands:
  {"action": "index", "kb_id": "...?", "directory_path": "...", ...}
  {"action": "index_files", "kb_id": "...?", "file_paths": [...], ...}
  {"action": "query", "kb_id": "...?", "query": "...", "k": int, ...}
  {"action": "clear_index", "kb_id": "...?"}
  {"action": "stats", "kb_id": "...?"}
  {"action": "health"}
  {"action": "list_docs", "kb_id": "...?"}
  {"action": "list_kbs"}
  {"action": "create_kb", "name": "...", "description": "..."}
  {"action": "delete_kb", "kb_id": "..."}
"""

import json
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))


def _resolve_kb_path(kb_id: str | None) -> str:
    """Resolve kb_id to a persist_directory. Migrates old data on first run.
    If kb_id is None, falls back to the first available KB.
    Mutates the global config so downstream VectorStoreManager picks up the path.
    """
    from ragify.core.kb_manager import KBManager
    from ragify.config import get_config

    manager = KBManager()
    manager.migrate_if_needed()

    if kb_id:
        kb = manager.get(kb_id)
        if kb is None:
            raise ValueError(f"知识库 '{kb_id}' 不存在")
    else:
        all_kbs = manager.list_all()
        if not all_kbs:
            raise ValueError("没有可用知识库，请先创建知识库")
        kb_id = all_kbs[0].id

    persist_dir = manager.get_persist_dir(kb_id)
    get_config().update("vectorstore.persist_directory", persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


# ── KB management handlers (no vectorstore config needed) ──────────

def handle_list_kbs(_data: dict) -> dict:
    from ragify.core.kb_manager import KBManager

    manager = KBManager()
    manager.migrate_if_needed()
    kbs = manager.list_all()
    kbs_out = []
    for kb in kbs:
        # derive doc count from the vectorstore in that KB's directory
        doc_count = 0
        try:
            from ragify.core.vectorstores import VectorStoreManager
            from ragify.config import get_config

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


def handle_create_kb(data: dict) -> dict:
    from ragify.core.kb_manager import KBManager

    name = data.get("name", "").strip()
    if not name:
        raise ValueError("知识库名称不能为空")
    description = data.get("description", "")

    manager = KBManager()
    manager.migrate_if_needed()
    kb = manager.create(name, description)
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "created_at": kb.created_at,
    }


def handle_delete_kb(data: dict) -> dict:
    from ragify.core.kb_manager import KBManager

    kb_id = data.get("kb_id", "")
    if not kb_id:
        raise ValueError("缺少 kb_id 参数")

    manager = KBManager()
    ok = manager.delete(kb_id)
    if not ok:
        raise ValueError(f"知识库 '{kb_id}' 不存在")
    return {"success": True}


# ── Core handlers (kb-aware via _resolve_kb_path) ──────────────────

def handle_index(data: dict) -> dict:
    _resolve_kb_path(data.get("kb_id"))
    from ragify.mcp import IndexingPipeline

    pipeline = IndexingPipeline()
    payload: dict[str, object] = {}
    if "directory_path" in data:
        payload["directory_path"] = data["directory_path"]
    if "file_paths" in data:
        payload["file_paths"] = data["file_paths"]
    if "clear_vectorstore" in data:
        payload["clear_vectorstore"] = data["clear_vectorstore"]

    result = pipeline.run(payload)
    return {"indexing_summary": result.get("indexing_summary", {})}


def handle_query(data: dict) -> dict:
    _resolve_kb_path(data.get("kb_id"))
    from ragify.mcp import QueryPipeline

    pipeline = QueryPipeline()
    result = pipeline.run({
        "query": data["query"],
        "k": data.get("k", 3),
        "score_threshold": data.get("score_threshold"),
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


def handle_clear(data: dict) -> dict:
    kb_id = data.get("kb_id")
    if kb_id:
        _resolve_kb_path(kb_id)
    from ragify.core.vectorstores import VectorStoreManager

    vm = VectorStoreManager()
    vm.clear()
    return {"success": True}


def handle_stats(data: dict) -> dict:
    if data.get("kb_id"):
        _resolve_kb_path(data["kb_id"])
    from ragify.core.vectorstores import VectorStoreManager
    from ragify.config import get_config

    cfg = get_config()
    vm = VectorStoreManager()
    return {
        "store_type": cfg.get("vectorstore.type", "unknown"),
        "collection_name": cfg.get("vectorstore.collection_name", ""),
        "persist_directory": cfg.get("vectorstore.persist_directory", ""),
        "doc_count": vm.get_document_count(),
    }


def handle_list_docs(data: dict) -> dict:
    if data.get("kb_id"):
        _resolve_kb_path(data["kb_id"])
    from ragify.core.vectorstores import VectorStoreManager

    vm = VectorStoreManager()
    sources = vm.get_sources()
    return {"documents": sources, "total": len(sources)}


def handle_health(_data: dict) -> dict:
    from ragify.config import get_config

    cfg = get_config()
    return {
        "status": "healthy",
        "version": cfg.get("base.version", "0.1.0"),
        "llm_provider": cfg.get("llm.provider", "unknown"),
        "vectorstore_type": cfg.get("vectorstore.type", "unknown"),
    }


HANDLERS = {
    "index": handle_index,
    "index_files": handle_index,
    "query": handle_query,
    "clear_index": handle_clear,
    "stats": handle_stats,
    "health": handle_health,
    "list_docs": handle_list_docs,
    "list_kbs": handle_list_kbs,
    "create_kb": handle_create_kb,
    "delete_kb": handle_delete_kb,
}


def main() -> None:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        action = data.get("action", "")

        if action not in HANDLERS:
            print(json.dumps({"error": f"Unknown action: {action}"}))
            sys.exit(1)

        result = HANDLERS[action](data)
        print(json.dumps(result, ensure_ascii=False, default=str))

    except Exception:
        print(json.dumps({
            "error": traceback.format_exc(),
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

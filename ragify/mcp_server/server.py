"""MCP (Model Context Protocol) stdio server for RAGify tools.

Implements JSON-RPC 2.0 per the MCP specification.
Entry point: python -m ragify.mcp_server.server
"""

import json
import sys
from typing import Any

from ..agentic.skills import SkillRegistry
from ..core.kb_manager import KBManager


def _list_tools() -> list[dict]:
    return [
        {
            "name": "ragify_query",
            "description": "Query a RAGify knowledge base and get an answer based on indexed documents.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to ask"},
                    "kb_id": {"type": "string", "description": "Knowledge base ID (optional)"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "ragify_list_kbs",
            "description": "List all available knowledge bases.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _list_resources() -> list[dict]:
    manager = KBManager()
    manager.migrate_if_needed()
    kbs = manager.list_all()
    resources: list[dict] = []
    for kb in kbs:
        resources.append({
            "uri": f"ragify://kb/{kb.id}",
            "name": kb.name,
            "description": kb.description or "",
            "mimeType": "application/json",
        })
    return resources


def _list_skills() -> list[dict]:
    registry = SkillRegistry()
    skills = []
    for skill in registry.get_all():
        skills.append({
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "keywords": skill.keywords,
        })
    return skills


def _handle_request(request: dict) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "tools/list":
        result = _list_tools()
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _call_tool(tool_name, arguments)
    elif method == "resources/list":
        result = _list_resources()
    elif method == "skills/list":
        result = _list_skills()
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _call_tool(name: str, arguments: dict) -> Any:
    if name == "ragify_query":
        query = arguments.get("query", "")
        kb_id = arguments.get("kb_id")
        try:
            from ..agentic.agent import AgenticRAG
            agent = AgenticRAG(kb_id=kb_id)
            result = agent.run(query)
            return result.get("response", "")
        except Exception as e:
            return f"Tool error: {e}"
    elif name == "ragify_list_kbs":
        manager = KBManager()
        manager.migrate_if_needed()
        return [{"id": kb.id, "name": kb.name} for kb in manager.list_all()]
    return {"error": f"Unknown tool: {name}"}


def run_mcp_server() -> None:
    """Run the MCP server on stdio (JSON-RPC 2.0, one request per line)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()

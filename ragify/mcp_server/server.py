"""MCP (Model Context Protocol) stdio server for RAGify tools.

Implements JSON-RPC 2.0 per the MCP specification.
Entry point: python -m ragify.mcp_server.server
"""

import json
import os
import sys
from typing import Any

import jwt as pyjwt

from ..agentic.skills import SkillRegistry
from ..core.kb_manager import KBManager
from ..core.security import decode_access_token
from ..core.tenant_manager import TenantManager
from ..core.user_manager import UserManager


def _resolve_mcp_tenant_id(
    user_manager: UserManager | None = None,
    tenant_manager: TenantManager | None = None,
    *,
    secret: str | None = None,
) -> str:
    """启动期解析当前 MCP 会话对应的 tenant_id。读 RAGIFY_MCP_TOKEN 环境变量
    （用户通过已有的 /api/auth/login 拿到的真实 JWT），解码拿 user_id，查用户
    是否存在，再取这个用户所属的第一个工作区。任何一步失败都直接抛异常让
    进程启动失败——不静默降级、不假装能继续跑。

    user_manager/tenant_manager/secret 三个参数只在测试里传（分别用临时
    数据库和固定密钥做确定性验证），生产代码路径永远不传，跟
    ragify/core/security.py 里 create_access_token/decode_access_token 的
    secret 参数是同一个"仅测试用"的设计思路。
    """
    user_manager = user_manager or UserManager()
    tenant_manager = tenant_manager or TenantManager()

    token = os.environ.get("RAGIFY_MCP_TOKEN")
    if not token:
        raise RuntimeError(
            "未设置 RAGIFY_MCP_TOKEN——MCP server 需要一个通过 /api/auth/login "
            "获取的有效登录凭证才能启动"
        )
    try:
        payload = decode_access_token(token, secret=secret)
        user_id = payload["sub"]
    except (pyjwt.PyJWTError, KeyError) as e:
        raise RuntimeError(f"RAGIFY_MCP_TOKEN 无效或已过期：{e}")

    user = user_manager.get_by_id(user_id)
    if user is None:
        raise RuntimeError("RAGIFY_MCP_TOKEN 对应的用户不存在")

    tenants = tenant_manager.list_tenants_for_user(user.id)
    if not tenants:
        raise RuntimeError(f"用户 {user.email} 目前不属于任何工作区，MCP server 无法启动")

    return tenants[0].id


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


def _list_resources(tenant_id: str) -> list[dict]:
    manager = KBManager()
    manager.migrate_json_if_needed()
    kbs = manager.list_all(tenant_id)
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


def _handle_request(request: dict, tenant_id: str) -> dict | None:
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "tools/list":
        result = _list_tools()
    elif method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _call_tool(tool_name, arguments, tenant_id)
    elif method == "resources/list":
        result = _list_resources(tenant_id)
    elif method == "skills/list":
        result = _list_skills()
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _call_tool(name: str, arguments: dict, tenant_id: str) -> Any:
    if name == "ragify_query":
        query = arguments.get("query", "")
        kb_id = arguments.get("kb_id")
        try:
            from ..api.dependencies import KB_LOCK, resolve_kb_path
            manager = KBManager()
            with KB_LOCK:
                resolve_kb_path(manager, kb_id, tenant_id)
            from ..agentic.agent import AgenticRAG
            agent = AgenticRAG(kb_id=kb_id)
            result = agent.run(query)
            return result.get("response", "")
        except Exception as e:
            return f"Tool error: {e}"
    elif name == "ragify_list_kbs":
        manager = KBManager()
        manager.migrate_json_if_needed()
        return [{"id": kb.id, "name": kb.name} for kb in manager.list_all(tenant_id)]
    return {"error": f"Unknown tool: {name}"}


def run_mcp_server() -> None:
    """Run the MCP server on stdio (JSON-RPC 2.0, one request per line)."""
    tenant_id = _resolve_mcp_tenant_id()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle_request(request, tenant_id)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()

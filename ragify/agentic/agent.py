"""Custom agent loop using OpenAI-compatible function calling.

Avoids LangChain 1.0.5's broken agent framework
(ImportError: cannot import name 'ExecutionInfo' from 'langgraph.runtime').
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import get_config
from ..core.language_models import LanguageModelManager
from ..core.vectorstores import VectorStoreManager

logger = logging.getLogger("ragify.agentic.agent")

MAX_ITERATIONS = 8

# ── OpenAI-compatible tool schema helpers ───────────────────────────


def _tool_to_function_def(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


# ── Default tools ───────────────────────────────────────────────────


def _retrieve_docs(query: str, kb_id: str | None = None) -> str:
    """Search the vector store for relevant documents."""
    vm = VectorStoreManager()
    cfg = get_config()
    k = cfg.get("retrieval.k", 3)
    docs = vm.similarity_search(query, k=k)
    if not docs:
        return "未找到相关文档。"
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[文档 {i}] 来源: {src}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    import ast
    import math
    import operator

    allowed_operators = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的操作符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的一元操作符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只允许简单函数调用")
            func_name = node.func.id
            allowed_funcs = {"abs": abs, "round": round, "max": max, "min": min, "sum": sum}
            if func_name in allowed_funcs:
                args = [_eval(a) for a in node.args]
                return allowed_funcs[func_name](*args)
            raise ValueError(f"不允许的函数调用: {func_name}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math":
                math_attrs = {
                    "sin": math.sin, "cos": math.cos, "tan": math.tan,
                    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
                    "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
                    "pi": math.pi, "e": math.e,
                }
                if node.attr in math_attrs:
                    return math_attrs[node.attr]
            raise ValueError("只允许访问 math 模块的属性")
        if isinstance(node, ast.Name):
            allowed_names = {"pi": math.pi, "e": math.e}
            if node.id in allowed_names:
                return allowed_names[node.id]
            raise ValueError(f"未定义的名称: {node.id}")
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return f"{expression} = {result}"
    except SyntaxError:
        return f"语法错误: 无法解析表达式 '{expression}'"
    except Exception as e:
        return f"计算失败: {str(e)}"


def get_agentic_tools(kb_id: str | None = None) -> list[dict]:
    """Assemble the default tool set available to the agent."""
    return [
        {
            "name": "retrieve_docs",
            "description": "从知识库中检索与查询相关的文档内容。用于查找知识库中的信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询字符串",
                    },
                },
                "required": ["query"],
            },
            "handler": lambda query: _retrieve_docs(query, kb_id),
        },
        {
            "name": "calculator",
            "description": "计算数学表达式。支持四则运算、math模块函数（sin, cos, sqrt, log等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '2 + 3 * 4' 或 'sqrt(16)'",
                    },
                },
                "required": ["expression"],
            },
            "handler": lambda expression: _calculator(expression),
        },
    ]


# ── Agent loop ──────────────────────────────────────────────────────


class AgenticRAG:
    """Lightweight agent using OpenAI-compatible function calling."""

    def __init__(
        self,
        kb_id: str | None = None,
        tools: list[dict] | None = None,
        max_iterations: int | None = None,
    ):
        cfg = get_config()
        self.kb_id = kb_id
        self.tools = tools or get_agentic_tools(kb_id)
        self.max_iterations = max_iterations or cfg.get("agentic.max_iterations", MAX_ITERATIONS)
        self.tool_map = {t["name"]: t["handler"] for t in self.tools}
        self.tool_schemas = [_tool_to_function_def(t) for t in self.tools]

        self.llm_manager = LanguageModelManager()
        self.llm = self.llm_manager.llm

    def run(self, query: str, chat_history: list[dict] | None = None) -> dict:
        tool_call_log: list[dict] = []
        sources: list[str] = []

        system_prompt = (
            "你是一个智能知识库助手，可以使用工具来检索信息和执行操作。\n"
            "当用户询问知识库中的内容时，必须先使用 retrieve_docs 工具检索文档。\n"
            "当需要进行数学计算时，使用 calculator 工具。\n"
            "基于检索结果给出准确、有引用来源的回答。"
        )

        messages: list = [SystemMessage(content=system_prompt)]

        if chat_history:
            for item in chat_history:
                if item.get("role") == "user":
                    messages.append(HumanMessage(content=item.get("content", "")))
                elif item.get("role") == "assistant":
                    messages.append(AIMessage(content=item.get("content", "")))

        messages.append(HumanMessage(content=query))

        for iteration in range(self.max_iterations):
            try:
                response = self.llm.invoke(
                    messages,
                    tools=self.tool_schemas if self.tool_schemas else None,
                )
            except Exception as e:
                logger.error(f"LLM call failed at iteration {iteration}: {e}")
                return {
                    "response": f"调用模型失败: {e}",
                    "tool_calls": tool_call_log,
                    "sources": sources,
                    "iterations": iteration,
                }

            content = getattr(response, "content", "") or ""
            raw_tool_calls = getattr(response, "tool_calls", []) or []

            if not raw_tool_calls and content:
                return {
                    "response": content,
                    "tool_calls": tool_call_log,
                    "sources": sources,
                    "iterations": iteration + 1,
                }

            if not raw_tool_calls:
                return {
                    "response": "智能体未能生成有效的回答。请重试。",
                    "tool_calls": tool_call_log,
                    "sources": sources,
                    "iterations": iteration + 1,
                }

            # Append the assistant message that requested tool calls
            messages.append(response)

            for tc in raw_tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                tc_id = tc.get("id", "")

                entry = {
                    "tool": name,
                    "input": args,
                    "output": "",
                    "iteration": iteration + 1,
                }

                if name in self.tool_map:
                    try:
                        result = self.tool_map[name](**args)
                        entry["output"] = str(result)
                    except Exception as e:
                        entry["output"] = f"工具执行错误: {e}"

                    if name == "retrieve_docs":
                        sources.append(str(args.get("query", "")))
                else:
                    entry["output"] = f"未知工具: {name}"

                tool_call_log.append(entry)

                # Append tool result as ToolMessage
                messages.append(ToolMessage(
                    content=entry["output"],
                    tool_call_id=tc_id,
                ))

        return {
            "response": "已达到最大工具调用次数，请尝试简化你的问题。",
            "tool_calls": tool_call_log,
            "sources": sources,
            "iterations": self.max_iterations,
        }

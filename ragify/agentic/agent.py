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
from .skills import SkillRegistry

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


def _retrieve_docs(query: str, kb_id: str | None = None, sink: list[dict] | None = None) -> str:
    """Search the vector store for relevant documents, filtered by the configured score threshold.

    Real source metadata and relevance scores are appended to `sink` (when given) so the
    caller can report accurate citations instead of guessing from the query text.
    """
    vm = VectorStoreManager()
    cfg = get_config()
    k = cfg.get("retrieval.k", 3)
    score_threshold = cfg.get("retrieval.score_threshold")
    results = vm.similarity_search_with_score(query, k=k, score_threshold=score_threshold)
    if not results:
        return (
            "未找到足够相关的文档（低于相关度阈值）。"
            "可以尝试更换关键词，或将问题拆解为更具体/更宽泛的表述后重新检索。"
        )
    parts: list[str] = []
    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[文档 {i}] 来源: {src} (相关度分数: {score:.3f})\n{doc.page_content}")
        if sink is not None:
            sink.append({"source": src, "score": float(score)})
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


def _index_directory(directory_path: str, clear_existing: bool = False) -> str:
    from ..agents.tools import IndexingTool
    return IndexingTool.index_directory(directory_path, clear_existing)


def _list_files(directory: str, pattern: str | None = None) -> str:
    from ..agents.tools import FileManagementTool
    return FileManagementTool.list_files(directory, pattern)


def _all_tool_defs(kb_id: str | None, sources_sink: list[dict] | None) -> dict[str, dict]:
    """Build the full catalogue of tools the agent can be given, keyed by name."""
    return {
        "retrieve_docs": {
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
            "handler": lambda query: _retrieve_docs(query, kb_id, sink=sources_sink),
        },
        "calculator": {
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
        "index_directory": {
            "name": "index_directory",
            "description": "将指定目录下的文档索引进知识库，用于用户要求补充/更新知识库内容时。",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {"type": "string", "description": "待索引的目录路径"},
                    "clear_existing": {
                        "type": "boolean",
                        "description": "是否在索引前清空现有向量库，默认 false",
                    },
                },
                "required": ["directory_path"],
            },
            "handler": lambda directory_path, clear_existing=False: _index_directory(
                directory_path, clear_existing
            ),
        },
        "list_files": {
            "name": "list_files",
            "description": "列出某个目录下的文件，用于探索可供索引的原始文档。",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "目录路径"},
                    "pattern": {"type": "string", "description": "可选的文件匹配模式，例如 '*.pdf'"},
                },
                "required": ["directory"],
            },
            "handler": lambda directory, pattern=None: _list_files(directory, pattern),
        },
    }


def get_agentic_tools(
    kb_id: str | None = None, sources_sink: list[dict] | None = None
) -> list[dict]:
    """Assemble the tool set available to the agent, per `agentic.tools` config."""
    cfg = get_config()
    enabled_names = cfg.get("agentic.tools") or ["retrieve_docs", "calculator"]
    catalogue = _all_tool_defs(kb_id, sources_sink)
    return [catalogue[name] for name in enabled_names if name in catalogue]


# ── Agent loop ──────────────────────────────────────────────────────


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    """Collapse repeated citations of the same source, keeping first-seen order."""
    seen: dict[str, dict] = {}
    for s in sources:
        seen.setdefault(s["source"], s)
    return list(seen.values())


BASE_SYSTEM_PROMPT = (
    "你是一个智能知识库助手，可以使用工具来检索信息和执行操作。\n"
    "当用户询问知识库中的内容时，必须先使用 retrieve_docs 工具检索文档。\n"
    "如果检索结果不相关或为空，尝试更换关键词、拆解问题后重新检索，而不是直接放弃。\n"
    "当需要进行数学计算时，使用 calculator 工具。\n"
    "基于检索结果给出准确、有引用来源的回答；如果检索结果不足以支撑答案，明确告知用户。"
)

REFLECTION_PROMPT = (
    "请检查你刚才的回答是否完全基于上面检索到的文档内容，且没有编造未出现过的信息。\n"
    "如果回答准确、有据可查，请原样重复该回答；如果存在不准确或缺乏依据的地方，请给出修正后的回答。\n"
    "直接输出最终回答本身，不要加任何解释或前缀。\n\n"
    f"你的草稿回答:\n{{draft}}"
)


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
        self.retrieved_sources: list[dict] = []
        self.tools = tools or get_agentic_tools(kb_id, sources_sink=self.retrieved_sources)
        self.max_iterations = max_iterations or cfg.get("agentic.max_iterations", MAX_ITERATIONS)
        self.tool_map = {t["name"]: t["handler"] for t in self.tools}
        self.tool_schemas = [_tool_to_function_def(t) for t in self.tools]
        self.enabled_skills = set(cfg.get("agentic.skills") or [])
        self.reflection_enabled = bool(cfg.get("agentic.reflection", False))

        self.llm_manager = LanguageModelManager()
        self.llm = self.llm_manager.llm

    def _build_system_prompt(self, query: str) -> str:
        """Match query keywords against enabled skills and fold their instructions in."""
        matched = [
            s for s in SkillRegistry().match(query)
            if not self.enabled_skills or s.name in self.enabled_skills
        ]
        if not matched:
            return BASE_SYSTEM_PROMPT
        skill_context = "\n".join(
            f"[已激活技能: {s.name}] {s.system_prompt}" for s in matched if s.system_prompt
        )
        return f"{BASE_SYSTEM_PROMPT}\n\n{skill_context}"

    def run(self, query: str, chat_history: list[dict] | None = None) -> dict:
        tool_call_log: list[dict] = []

        system_prompt = self._build_system_prompt(query)

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
                    "sources": _dedupe_sources(self.retrieved_sources),
                    "iterations": iteration,
                }

            content = getattr(response, "content", "") or ""
            raw_tool_calls = getattr(response, "tool_calls", []) or []

            if not raw_tool_calls and content:
                if self.reflection_enabled and self.retrieved_sources:
                    content = self._reflect(messages, content)
                return {
                    "response": content,
                    "tool_calls": tool_call_log,
                    "sources": _dedupe_sources(self.retrieved_sources),
                    "iterations": iteration + 1,
                }

            if not raw_tool_calls:
                return {
                    "response": "智能体未能生成有效的回答。请重试。",
                    "tool_calls": tool_call_log,
                    "sources": _dedupe_sources(self.retrieved_sources),
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
            "sources": _dedupe_sources(self.retrieved_sources),
            "iterations": self.max_iterations,
        }

    def _reflect(self, messages: list, draft_answer: str) -> str:
        """One extra LLM pass that checks the draft answer against retrieved context.

        Gated behind `agentic.reflection` since it doubles the LLM calls for the final
        turn — worth it for answer quality, but adds latency/cost, so it defaults to off.
        """
        reflect_messages = messages + [
            HumanMessage(content=REFLECTION_PROMPT.format(draft=draft_answer))
        ]
        try:
            response = self.llm.invoke(reflect_messages)
            revised = (getattr(response, "content", "") or "").strip()
            return revised or draft_answer
        except Exception as e:
            logger.warning(f"Reflection step failed, keeping draft answer: {e}")
            return draft_answer

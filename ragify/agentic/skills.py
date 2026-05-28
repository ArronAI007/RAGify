"""Skills registry for agentic RAG — pluggable capabilities activated by keyword matching."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Skill:
    name: str
    description: str
    version: str = "1.0.0"
    keywords: list[str] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    system_prompt: str = ""


class SkillRegistry:
    """Simple registry for discoverable, keyword-matchable skills."""

    _instance: "SkillRegistry | None" = None

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._register_builtins()

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
            cls._instance._register_builtins()
        return cls._instance

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        return list(self._skills.values())

    def match(self, query: str) -> list[Skill]:
        """Return skills whose keywords appear in the query, ranked by overlap count."""
        query_lower = query.lower()
        scored: list[tuple[int, Skill]] = []
        for skill in self._skills.values():
            count = sum(1 for kw in skill.keywords if kw.lower() in query_lower)
            if count > 0:
                scored.append((count, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def _register_builtins(self) -> None:
        self.register(Skill(
            name="summarization",
            description="Summarize documents or text content",
            version="1.0.0",
            keywords=["总结", "概括", "摘要", "简述", "summarize", "summary", "概述"],
            system_prompt=(
                "你是一个专业的文本摘要助手。请对检索到的内容进行简洁、准确的总结。"
                "保留关键信息，省略冗余细节。"
            ),
        ))
        self.register(Skill(
            name="data_extraction",
            description="Extract structured data from documents",
            version="1.0.0",
            keywords=["提取", "抽取", "数据", "表格", "列表", "extract", "data", "结构化"],
            system_prompt=(
                "你是一个专业的数据提取助手。请从检索到的文档中提取结构化信息，"
                "以表格或列表形式呈现。标注每个数据项的来源。"
            ),
        ))

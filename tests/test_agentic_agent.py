#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgenticRAG 专项测试
验证检索分数阈值过滤、真实来源追踪、技能匹配注入 system prompt、
以及可配置的反思(reflection)步骤。
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.agentic.agent import AgenticRAG, _dedupe_sources, _retrieve_docs


def _fake_config(overrides: dict | None = None):
    values = {
        "retrieval.k": 3,
        "retrieval.score_threshold": 0.7,
        "agentic.max_iterations": 8,
        "agentic.tools": ["retrieve_docs", "calculator"],
        "agentic.skills": ["summarization", "data_extraction"],
        "agentic.reflection": False,
    }
    values.update(overrides or {})
    cfg = MagicMock()
    cfg.get.side_effect = lambda key, default=None: values.get(key, default)
    return cfg


def _mock_doc(source: str, content: str = "内容"):
    doc = MagicMock()
    doc.metadata = {"source": source}
    doc.page_content = content
    return doc


class TestRetrieveDocs(unittest.TestCase):
    """检索质量：分数阈值过滤 + 真实来源写入 sink"""

    @patch("ragify.agentic.agent.get_config")
    @patch("ragify.agentic.agent.VectorStoreManager")
    def test_uses_score_threshold_from_config(self, mock_vsm_cls, mock_get_config):
        mock_get_config.return_value = _fake_config({"retrieval.score_threshold": 0.5})
        mock_vsm = MagicMock()
        mock_vsm.similarity_search_with_score.return_value = [(_mock_doc("a.txt"), 0.9)]
        mock_vsm_cls.return_value = mock_vsm

        _retrieve_docs("问题")

        mock_vsm.similarity_search_with_score.assert_called_once_with(
            "问题", k=3, score_threshold=0.5
        )

    @patch("ragify.agentic.agent.get_config")
    @patch("ragify.agentic.agent.VectorStoreManager")
    def test_sink_receives_real_source_and_score(self, mock_vsm_cls, mock_get_config):
        mock_get_config.return_value = _fake_config()
        mock_vsm = MagicMock()
        mock_vsm.similarity_search_with_score.return_value = [
            (_mock_doc("docs/a.txt"), 0.92),
            (_mock_doc("docs/b.txt"), 0.81),
        ]
        mock_vsm_cls.return_value = mock_vsm

        sink: list[dict] = []
        text = _retrieve_docs("问题", sink=sink)

        self.assertEqual(sink, [
            {"source": "docs/a.txt", "score": 0.92},
            {"source": "docs/b.txt", "score": 0.81},
        ])
        self.assertIn("docs/a.txt", text)

    @patch("ragify.agentic.agent.get_config")
    @patch("ragify.agentic.agent.VectorStoreManager")
    def test_empty_results_returns_guidance_without_touching_sink(self, mock_vsm_cls, mock_get_config):
        mock_get_config.return_value = _fake_config()
        mock_vsm = MagicMock()
        mock_vsm.similarity_search_with_score.return_value = []
        mock_vsm_cls.return_value = mock_vsm

        sink: list[dict] = []
        text = _retrieve_docs("问题", sink=sink)

        self.assertEqual(sink, [])
        self.assertIn("未找到", text)


class TestDedupeSources(unittest.TestCase):
    def test_keeps_first_occurrence_per_source(self):
        sources = [
            {"source": "a.txt", "score": 0.9},
            {"source": "b.txt", "score": 0.8},
            {"source": "a.txt", "score": 0.5},
        ]
        result = _dedupe_sources(sources)
        self.assertEqual(result, [
            {"source": "a.txt", "score": 0.9},
            {"source": "b.txt", "score": 0.8},
        ])


class TestAgenticRAGRun(unittest.TestCase):
    """完整 run() 循环：真实 sources、技能匹配、反思开关"""

    def _make_agent(self, mock_get_config, mock_vsm_cls, mock_lmm_cls, config_overrides=None):
        mock_get_config.return_value = _fake_config(config_overrides)
        mock_vsm = MagicMock()
        mock_vsm.similarity_search_with_score.return_value = [(_mock_doc("kb/doc1.txt"), 0.88)]
        mock_vsm_cls.return_value = mock_vsm

        mock_llm = MagicMock()
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.llm = mock_llm
        mock_lmm_cls.return_value = mock_lmm_instance
        return mock_llm

    @patch("ragify.agentic.agent.LanguageModelManager")
    @patch("ragify.agentic.agent.VectorStoreManager")
    @patch("ragify.agentic.agent.get_config")
    def test_sources_reflect_real_retrieved_metadata(self, mock_get_config, mock_vsm_cls, mock_lmm_cls):
        mock_llm = self._make_agent(mock_get_config, mock_vsm_cls, mock_lmm_cls)

        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "retrieve_docs", "args": {"query": "什么是RAG"}, "id": "call_1"}
        ]
        final_response = MagicMock()
        final_response.content = "RAG 是检索增强生成。"
        final_response.tool_calls = []
        mock_llm.invoke.side_effect = [tool_call_response, final_response]

        agent = AgenticRAG(kb_id="kb1")
        result = agent.run("什么是RAG")

        self.assertEqual(result["sources"], [{"source": "kb/doc1.txt", "score": 0.88}])
        self.assertEqual(result["response"], "RAG 是检索增强生成。")

    @patch("ragify.agentic.agent.LanguageModelManager")
    @patch("ragify.agentic.agent.VectorStoreManager")
    @patch("ragify.agentic.agent.get_config")
    def test_skill_match_is_folded_into_system_prompt(self, mock_get_config, mock_vsm_cls, mock_lmm_cls):
        mock_llm = self._make_agent(mock_get_config, mock_vsm_cls, mock_lmm_cls)
        final_response = MagicMock()
        final_response.content = "摘要内容"
        final_response.tool_calls = []
        mock_llm.invoke.return_value = final_response

        agent = AgenticRAG(kb_id="kb1")
        agent.run("请帮我总结一下这份文档")

        sent_messages = mock_llm.invoke.call_args[0][0]
        system_prompt = sent_messages[0].content
        self.assertIn("summarization", system_prompt)

    @patch("ragify.agentic.agent.LanguageModelManager")
    @patch("ragify.agentic.agent.VectorStoreManager")
    @patch("ragify.agentic.agent.get_config")
    def test_reflection_disabled_by_default_skips_extra_llm_call(self, mock_get_config, mock_vsm_cls, mock_lmm_cls):
        mock_llm = self._make_agent(mock_get_config, mock_vsm_cls, mock_lmm_cls)
        final_response = MagicMock()
        final_response.content = "答案"
        final_response.tool_calls = []
        mock_llm.invoke.return_value = final_response

        agent = AgenticRAG(kb_id="kb1")
        agent.run("普通问题")

        self.assertEqual(mock_llm.invoke.call_count, 1)

    @patch("ragify.agentic.agent.LanguageModelManager")
    @patch("ragify.agentic.agent.VectorStoreManager")
    @patch("ragify.agentic.agent.get_config")
    def test_reflection_enabled_makes_extra_call_when_sources_exist(self, mock_get_config, mock_vsm_cls, mock_lmm_cls):
        mock_llm = self._make_agent(
            mock_get_config, mock_vsm_cls, mock_lmm_cls, config_overrides={"agentic.reflection": True}
        )

        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "retrieve_docs", "args": {"query": "问题"}, "id": "call_1"}
        ]
        draft_response = MagicMock()
        draft_response.content = "草稿答案"
        draft_response.tool_calls = []
        reflection_response = MagicMock()
        reflection_response.content = "校验后的答案"
        mock_llm.invoke.side_effect = [tool_call_response, draft_response, reflection_response]

        agent = AgenticRAG(kb_id="kb1")
        result = agent.run("问题")

        self.assertEqual(mock_llm.invoke.call_count, 3)
        self.assertEqual(result["response"], "校验后的答案")


if __name__ == "__main__":
    unittest.main()

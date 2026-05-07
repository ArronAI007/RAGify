#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAgent 专项测试
验证 RAGAgent 初始化、invoke 和工具行为。
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.agents.rag_agent import RAGAgent, MultiModalRAGAgent, PipelineAgent
from ragify.agents.tools import create_custom_tool


class TestRAGAgent(unittest.TestCase):
    """测试 RAGAgent"""

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_initialization(self, mock_vsm, mock_lmm, mock_qp):
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_vsm_instance = MagicMock()
        mock_vsm.return_value = mock_vsm_instance

        mock_qp_instance = MagicMock()
        mock_qp.return_value = mock_qp_instance

        agent = RAGAgent(name="test_rag")
        self.assertEqual(agent.name, "test_rag")
        self.assertTrue(len(agent.tools) >= 2)  # rag_query + vectorstore_info

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_invoke_returns_response(self, mock_vsm, mock_lmm, mock_qp):
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_vsm_instance = MagicMock()
        mock_vsm.return_value = mock_vsm_instance

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "测试响应",
            "retrieved_documents": [],
        }
        mock_qp.return_value = mock_pipeline

        agent = RAGAgent(name="test_rag")
        result = agent.invoke("测试查询")

        self.assertIn("query", result)
        self.assertIn("response", result)
        self.assertEqual(result["query"], "测试查询")

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_rag_query_with_sources(self, mock_vsm, mock_lmm, mock_qp):
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "test.txt", "retrieval_score": 0.95}

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "答案",
            "retrieved_documents": [mock_doc],
        }
        mock_qp.return_value = mock_pipeline

        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_vsm_instance = MagicMock()
        mock_vsm.return_value = mock_vsm_instance

        agent = RAGAgent(name="test_rag")
        result = agent.invoke("查询")
        self.assertIn("response", result)

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_vectorstore_info_tool(self, mock_vsm, mock_lmm, mock_qp):
        mock_vsm_instance = MagicMock()
        mock_vsm_instance.get_document_count.return_value = 42
        mock_vsm_instance.vectorstore_type = "chromadb"
        mock_vsm.return_value = mock_vsm_instance

        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_qp_instance = MagicMock()
        mock_qp.return_value = mock_qp_instance

        agent = RAGAgent(name="test_rag")
        info = agent._vectorstore_info()
        self.assertIn("42", info)
        self.assertIn("chromadb", info)


class TestMultiModalRAGAgent(unittest.TestCase):
    """测试 MultiModalRAGAgent"""

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.mcp.MultiModalQueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_initialization(self, mock_vsm, mock_lmm, mock_mqp, mock_qp):
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_vsm_instance = MagicMock()
        mock_vsm.return_value = mock_vsm_instance

        mock_mqp_instance = MagicMock()
        mock_mqp.return_value = mock_mqp_instance

        mock_qp_instance = MagicMock()
        mock_qp.return_value = mock_qp_instance

        agent = MultiModalRAGAgent(name="mm_test")
        self.assertEqual(agent.name, "mm_test")
        tool_names = [t.name for t in agent.tools]
        self.assertIn("multimodal_query", tool_names)

    @patch("ragify.agents.rag_agent.QueryPipeline")
    @patch("ragify.mcp.MultiModalQueryPipeline")
    @patch("ragify.agents.rag_agent.LanguageModelManager")
    @patch("ragify.agents.rag_agent.VectorStoreManager")
    def test_multimodal_query_with_images(self, mock_vsm, mock_lmm, mock_mqp, mock_qp):
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = MagicMock()
        mock_lmm.return_value = mock_lmm_instance

        mock_vsm_instance = MagicMock()
        mock_vsm.return_value = mock_vsm_instance

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"response": "多模态答案"}
        mock_mqp.return_value = mock_pipeline

        mock_qp_instance = MagicMock()
        mock_qp.return_value = mock_qp_instance

        agent = MultiModalRAGAgent(name="mm_test")
        result = agent.invoke("描述图像", image_urls=["http://example.com/img.jpg"])
        self.assertIn("response", result)


class TestPipelineAgent(unittest.TestCase):
    """测试 PipelineAgent"""

    @patch("ragify.agents.rag_agent.QueryPipeline")
    def test_invoke(self, mock_qp):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "流水线答案",
            "retrieved_documents": [],
            "query_summary": {},
        }
        mock_qp.return_value = mock_pipeline

        agent = PipelineAgent(name="pipe_test")
        result = agent.invoke("查询")
        self.assertEqual(result["response"], "流水线答案")
        self.assertEqual(result["query"], "查询")


if __name__ == "__main__":
    unittest.main()

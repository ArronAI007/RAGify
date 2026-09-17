#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/query 和 /api/query/agentic 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestQueryRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.manager.create("默认知识库")
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_query_without_available_kb_returns_400(self):
        # 删掉 setUp 里创建的那个默认知识库，制造"没有可用知识库"的情况
        kbs = self.manager.list_all()
        self.manager.delete(kbs[0].id)

        res = self.client.post("/api/query", json={"query": "test"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    @patch("ragify.api.routers.query.QueryPipeline")
    def test_query_reshapes_pipeline_result(self, mock_pipeline_cls):
        mock_doc = MagicMock()
        mock_doc.page_content = "内容"
        mock_doc.metadata = {"source": "a.txt", "file_type": "txt", "retrieval_score": 0.9}

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "答案",
            "response_generated": True,
            "retrieved_documents": [mock_doc],
            "query_summary": {"documents_retrieved": 1},
        }
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post("/api/query", json={"query": "什么是RAG", "k": 3})

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["response"], "答案")
        self.assertEqual(body["retrieved_documents"][0]["metadata"]["source"], "a.txt")
        mock_pipeline.run.assert_called_once_with({
            "query": "什么是RAG", "k": 3, "score_threshold": None,
        })

    @patch("ragify.api.routers.query.AgenticRAG")
    def test_agentic_query_delegates_to_agent(self, mock_agent_cls):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "response": "答案", "tool_calls": [], "sources": [], "iterations": 1,
        }
        mock_agent_cls.return_value = mock_agent

        res = self.client.post("/api/query/agentic", json={"query": "问题"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["response"], "答案")
        mock_agent.run.assert_called_once_with("问题", chat_history=None)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/query 和 .../query/agentic 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base, TenantAccountJoinRow
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
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        tenant_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        self.tenant_id = tenant_res.json()["id"]
        self.manager.create("默认知识库", "", self.tenant_id)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _register(self, email: str, name: str) -> str:
        res = self.client.post("/api/auth/register", json={
            "email": email, "password": "password123", "name": name,
        })
        return res.json()["access_token"]

    def _auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def _add_member(self, email: str, name: str, role: str) -> str:
        token = self._register(email, name)
        user_id = self.user_manager.get_by_email(email).id
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=self.tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()
        return token

    def test_query_without_available_kb_returns_400(self):
        kbs = self.manager.list_all(self.tenant_id)
        self.manager.delete(kbs[0].id, self.tenant_id)

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    def test_agentic_query_without_available_kb_returns_400(self):
        kbs = self.manager.list_all(self.tenant_id)
        self.manager.delete(kbs[0].id, self.tenant_id)

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query/agentic", json={"query": "test"}, headers=self._auth(self.owner_token)
        )
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

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query",
            json={"query": "什么是RAG", "k": 3},
            headers=self._auth(self.owner_token),
        )

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

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query/agentic",
            json={"query": "问题"},
            headers=self._auth(self.owner_token),
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["response"], "答案")
        mock_agent.run.assert_called_once_with("问题", chat_history=None)

    def test_no_token_rejected(self):
        res = self.client.post(f"/api/tenants/{self.tenant_id}/query", json={"query": "test"})
        self.assertEqual(res.status_code, 401)

    def test_non_member_cannot_query(self):
        other_token = self._register("other@example.com", "Other")
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(other_token)
        )
        self.assertEqual(res.status_code, 403)

    @patch("ragify.api.routers.query.QueryPipeline")
    def test_normal_member_can_query(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {
            "response": "答案", "response_generated": True,
            "retrieved_documents": [], "query_summary": {},
        }
        mock_pipeline_cls.return_value = mock_pipeline
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")

        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/query", json={"query": "test"}, headers=self._auth(normal_token)
        )
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

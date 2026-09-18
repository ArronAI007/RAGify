#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/{index,stats,documents,chunks} 路由测试。"""

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


class TestDocumentsRoutes(unittest.TestCase):
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
        self.kb = self.manager.create("默认知识库", "", self.tenant_id)

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

    def _p(self, path: str) -> str:
        return f"/api/tenants/{self.tenant_id}{path}"

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_directory_path(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 2}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post(self._p("/index"), json={
            "directory_path": "/some/dir", "kb_id": self.kb.id,
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["indexing_summary"]["total_documents_indexed"], 2)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], "/some/dir")
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_file_paths_does_not_default_clear_vectorstore(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 1}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post(self._p("/index"), json={
            "file_paths": ["/some/file.txt"], "kb_id": self.kb.id,
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["file_paths"], ["/some/file.txt"])
        self.assertNotIn("clear_vectorstore", called_payload)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_falls_back_to_kb_data_dir_when_no_path_or_files_given(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 0}}
        mock_pipeline_cls.return_value = mock_pipeline

        kb_data_dir = Path(self.tmp_dir) / "project_root_stub" / "data" / self.kb.id

        with patch("ragify.api.routers.documents.PROJECT_ROOT", str(Path(self.tmp_dir) / "project_root_stub")):
            kb_data_dir.mkdir(parents=True, exist_ok=True)
            res = self.client.post(
                self._p("/index"), json={"kb_id": self.kb.id}, headers=self._auth(self.owner_token)
            )

        self.assertEqual(res.status_code, 200)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], str(kb_data_dir))
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_clear_index(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request(
            "DELETE", self._p("/index"), json={"kb_id": self.kb.id}, headers=self._auth(self.owner_token)
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})
        mock_vsm.clear.assert_called_once()

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_get_stats(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_document_count.return_value = 5
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(self._p(f"/stats?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["doc_count"], 5)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_documents(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_sources.return_value = [{"name": "a.txt", "source": "a.txt"}]
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total"], 1)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_delete_document(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.delete_by_source.return_value = 3
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", self._p("/documents"), json={
            "kb_id": self.kb.id, "source": "nonexistent.txt",
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["chunks_removed"], 3)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_chunks_requires_source(self, mock_vsm_cls):
        res = self.client.get(self._p(f"/chunks?kb_id={self.kb.id}"), headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 422)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_update_chunk(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.update_chunk_content.return_value = True
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.put(self._p("/chunks"), json={
            "kb_id": self.kb.id, "chunk_id": "c1", "content": "新内容",
        }, headers=self._auth(self.owner_token))

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_get_stats_without_kb_returns_400_when_no_kbs_exist(self):
        self.manager.delete(self.kb.id, self.tenant_id)
        res = self.client.get(self._p("/stats"), headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])

    def test_no_token_rejected(self):
        res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"))
        self.assertEqual(res.status_code, 401)

    def test_normal_member_can_read_but_cannot_upload_or_delete(self):
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")

        read_res = self.client.get(self._p(f"/documents?kb_id={self.kb.id}"), headers=self._auth(normal_token))
        self.assertEqual(read_res.status_code, 200)

        upload_res = self.client.post(
            self._p("/index"), json={"kb_id": self.kb.id, "file_paths": ["/x.txt"]},
            headers=self._auth(normal_token),
        )
        self.assertEqual(upload_res.status_code, 403)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_dataset_operator_can_upload_documents(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 1}}
        mock_pipeline_cls.return_value = mock_pipeline
        operator_token = self._add_member("operator@example.com", "Operator", "DATASET_OPERATOR")

        res = self.client.post(
            self._p("/index"), json={"kb_id": self.kb.id, "file_paths": ["/x.txt"]},
            headers=self._auth(operator_token),
        )
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

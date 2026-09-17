#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/index、/api/stats、/api/documents、/api/chunks 路由测试。"""

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


class TestDocumentsRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        self.kb = self.manager.create("默认知识库")
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("ragify.api.routers.documents.IndexingPipeline")
    def test_index_with_directory_path(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"indexing_summary": {"total_documents_indexed": 2}}
        mock_pipeline_cls.return_value = mock_pipeline

        res = self.client.post("/api/index", json={
            "directory_path": "/some/dir", "kb_id": self.kb.id,
        })

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

        res = self.client.post("/api/index", json={
            "file_paths": ["/some/file.txt"], "kb_id": self.kb.id,
        })

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
            res = self.client.post("/api/index", json={"kb_id": self.kb.id})

        self.assertEqual(res.status_code, 200)
        called_payload = mock_pipeline.run.call_args[0][0]
        self.assertEqual(called_payload["directory_path"], str(kb_data_dir))
        self.assertTrue(called_payload["clear_vectorstore"])

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_clear_index(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", "/api/index", json={"kb_id": self.kb.id})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})
        mock_vsm.clear.assert_called_once()

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_get_stats(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_document_count.return_value = 5
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(f"/api/stats?kb_id={self.kb.id}")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["doc_count"], 5)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_documents(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.get_sources.return_value = [{"name": "a.txt", "source": "a.txt"}]
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.get(f"/api/documents?kb_id={self.kb.id}")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["total"], 1)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_delete_document(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.delete_by_source.return_value = 3
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.request("DELETE", "/api/documents", json={
            "kb_id": self.kb.id, "source": "nonexistent.txt",
        })

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["chunks_removed"], 3)

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_list_chunks_requires_source(self, mock_vsm_cls):
        res = self.client.get(f"/api/chunks?kb_id={self.kb.id}")
        self.assertEqual(res.status_code, 422)  # FastAPI 校验 source 是必填 query 参数

    @patch("ragify.api.routers.documents.VectorStoreManager")
    def test_update_chunk(self, mock_vsm_cls):
        mock_vsm = MagicMock()
        mock_vsm.update_chunk_content.return_value = True
        mock_vsm_cls.return_value = mock_vsm

        res = self.client.put("/api/chunks", json={
            "kb_id": self.kb.id, "chunk_id": "c1", "content": "新内容",
        })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_get_stats_without_kb_returns_400_when_no_kbs_exist(self):
        # 删掉 setUp 里创建的默认知识库，验证在"没有任何 kb_id、也没有任何知识库"
        # 的情况下，所有会调用 resolve_kb_path 的接口都干净地返回 400，
        # 而不是悄悄用全局配置里可能是别的请求残留下来的 persist_directory。
        self.manager.delete(self.kb.id)
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 400)
        self.assertIn("没有可用知识库", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()

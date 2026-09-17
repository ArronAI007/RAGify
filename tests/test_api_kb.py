#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/kb 路由测试：建/查/删知识库。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestKBRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(
            database_url=self.db_url,
            vectorstore_dir=Path(self.tmp_dir) / "vectorstore",
        )
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_list_kbs_empty(self):
        res = self.client.get("/api/kb")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"knowledge_bases": []})

    def test_create_and_list_kb(self):
        res = self.client.post("/api/kb", json={"name": "测试库", "description": "desc"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "测试库")

        res = self.client.get("/api/kb")
        kbs = res.json()["knowledge_bases"]
        self.assertEqual(len(kbs), 1)
        self.assertEqual(kbs[0]["name"], "测试库")
        self.assertEqual(kbs[0]["doc_count"], 0)

    def test_create_kb_empty_name_rejected(self):
        res = self.client.post("/api/kb", json={"name": "   "})
        self.assertEqual(res.status_code, 400)

    def test_create_kb_duplicate_name_rejected(self):
        self.client.post("/api/kb", json={"name": "重复"})
        res = self.client.post("/api/kb", json={"name": "重复"})
        self.assertEqual(res.status_code, 400)

    def test_delete_kb(self):
        create_res = self.client.post("/api/kb", json={"name": "待删除"})
        kb_id = create_res.json()["id"]
        res = self.client.delete(f"/api/kb/{kb_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_delete_missing_kb_returns_404(self):
        res = self.client.delete("/api/kb/does-not-exist")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()

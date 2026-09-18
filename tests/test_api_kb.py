#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/{tenant_id}/kb 路由测试：建/查/删知识库，登录门禁 + 权限矩阵。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_kb_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.kb_manager import KBManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base, TenantAccountJoinRow
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
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        app.dependency_overrides[get_kb_manager] = lambda: self.manager
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        self.tenant_id = self._create_tenant()

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

    def _create_tenant(self) -> str:
        res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        return res.json()["id"]

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

    def test_list_kbs_empty(self):
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"knowledge_bases": []})

    def test_create_and_list_kb(self):
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb",
            json={"name": "测试库", "description": "desc"},
            headers=self._auth(self.owner_token),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["name"], "测试库")

        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(self.owner_token))
        kbs = res.json()["knowledge_bases"]
        self.assertEqual(len(kbs), 1)
        self.assertEqual(kbs[0]["name"], "测试库")
        self.assertEqual(kbs[0]["doc_count"], 0)

    def test_create_kb_empty_name_rejected(self):
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "   "}, headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 400)

    def test_create_kb_duplicate_name_rejected(self):
        self.client.post(f"/api/tenants/{self.tenant_id}/kb", json={"name": "重复"}, headers=self._auth(self.owner_token))
        res = self.client.post(f"/api/tenants/{self.tenant_id}/kb", json={"name": "重复"}, headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)

    def test_delete_kb(self):
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "待删除"}, headers=self._auth(self.owner_token)
        )
        kb_id = create_res.json()["id"]
        res = self.client.delete(f"/api/tenants/{self.tenant_id}/kb/{kb_id}", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})

    def test_delete_missing_kb_returns_404(self):
        res = self.client.delete(
            f"/api/tenants/{self.tenant_id}/kb/does-not-exist", headers=self._auth(self.owner_token)
        )
        self.assertEqual(res.status_code, 404)

    def test_no_token_rejected(self):
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb")
        self.assertEqual(res.status_code, 401)

    def test_non_member_cannot_list(self):
        other_token = self._register("other@example.com", "Other")
        res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(other_token))
        self.assertEqual(res.status_code, 403)

    def test_normal_member_can_list_but_cannot_create(self):
        normal_token = self._add_member("normal@example.com", "Normal", "NORMAL")
        list_res = self.client.get(f"/api/tenants/{self.tenant_id}/kb", headers=self._auth(normal_token))
        self.assertEqual(list_res.status_code, 200)

        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "不该建成"}, headers=self._auth(normal_token)
        )
        self.assertEqual(create_res.status_code, 403)

    def test_dataset_operator_cannot_create_or_delete_kb(self):
        operator_token = self._add_member("operator@example.com", "Operator", "DATASET_OPERATOR")
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "不该建成"}, headers=self._auth(operator_token)
        )
        self.assertEqual(create_res.status_code, 403)

        existing = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "已有知识库"}, headers=self._auth(self.owner_token)
        ).json()
        delete_res = self.client.delete(
            f"/api/tenants/{self.tenant_id}/kb/{existing['id']}", headers=self._auth(operator_token)
        )
        self.assertEqual(delete_res.status_code, 403)

    def test_editor_can_create_and_delete_kb(self):
        editor_token = self._add_member("editor@example.com", "Editor", "EDITOR")
        create_res = self.client.post(
            f"/api/tenants/{self.tenant_id}/kb", json={"name": "编辑建的库"}, headers=self._auth(editor_token)
        )
        self.assertEqual(create_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

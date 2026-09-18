#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/tenants/* 路由测试（工作区建/查、成员管理、邀请管理）。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_invitation_manager, get_tenant_manager, get_user_manager
from ragify.api.main import app
from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestTenantRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        self.invitation_manager = InvitationManager(database_url=self.db_url)
        app.dependency_overrides[get_user_manager] = lambda: self.user_manager
        app.dependency_overrides[get_tenant_manager] = lambda: self.tenant_manager
        app.dependency_overrides[get_invitation_manager] = lambda: self.invitation_manager
        self.client = TestClient(app)

        self.owner_token = self._register("owner@example.com", "Owner")
        self.other_token = self._register("other@example.com", "Other")

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

    def test_create_tenant_makes_creator_owner(self):
        res = self.client.post("/api/tenants", json={"name": "测试工作区"}, headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        tenant_id = res.json()["id"]

        members_res = self.client.get(f"/api/tenants/{tenant_id}/members", headers=self._auth(self.owner_token))
        self.assertEqual(members_res.status_code, 200)
        roles = {m["role"] for m in members_res.json()}
        self.assertEqual(roles, {"OWNER"})

    def test_list_my_tenants(self):
        self.client.post("/api/tenants", json={"name": "工作区A"}, headers=self._auth(self.owner_token))
        res = self.client.get("/api/tenants", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)

    def test_non_member_cannot_list_members(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.get(f"/api/tenants/{tenant_id}/members", headers=self._auth(self.other_token))
        self.assertEqual(res.status_code, 403)

    def test_update_member_role_requires_owner_or_admin(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        # other 不是成员，改角色应该 403
        other_user_id = self.user_manager.get_by_email("other@example.com").id
        res = self.client.patch(
            f"/api/tenants/{tenant_id}/members/{other_user_id}",
            json={"role": "NORMAL"},
            headers=self._auth(self.other_token),
        )
        self.assertEqual(res.status_code, 403)

    def test_delete_tenant_requires_owner(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.delete(f"/api/tenants/{tenant_id}", headers=self._auth(self.other_token))
        self.assertEqual(res.status_code, 403)

        res = self.client.delete(f"/api/tenants/{tenant_id}", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 200)

    def test_sole_owner_cannot_leave(self):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.post(f"/api/tenants/{tenant_id}/leave", headers=self._auth(self.owner_token))
        self.assertEqual(res.status_code, 400)

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_create_invitation_sends_email(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]

        res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "invitee@example.com", "role": "NORMAL"},
            headers=self._auth(self.owner_token),
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "invitee@example.com")
        mock_send.assert_called_once()

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_admin_cannot_invite_as_admin(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]
        other_user_id = self.user_manager.get_by_email("other@example.com").id
        # 手工把 other 加成 ADMIN，验证 ADMIN 邀请 ADMIN 会被拒绝
        from ragify.db.models import TenantAccountJoinRow
        import uuid
        from datetime import datetime, timezone
        with self.tenant_manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id=other_user_id,
                role="ADMIN", created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()

        res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "x@example.com", "role": "ADMIN"},
            headers=self._auth(self.other_token),
        )
        self.assertEqual(res.status_code, 403)
        mock_send.assert_not_called()

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def test_list_and_revoke_invitations(self, mock_send):
        create_res = self.client.post("/api/tenants", json={"name": "工作区"}, headers=self._auth(self.owner_token))
        tenant_id = create_res.json()["id"]
        invite_res = self.client.post(
            f"/api/tenants/{tenant_id}/invitations",
            json={"email": "invitee@example.com", "role": "NORMAL"},
            headers=self._auth(self.owner_token),
        )
        invitation_id = invite_res.json()["id"]

        list_res = self.client.get(f"/api/tenants/{tenant_id}/invitations", headers=self._auth(self.owner_token))
        self.assertEqual(len(list_res.json()), 1)

        revoke_res = self.client.delete(
            f"/api/tenants/{tenant_id}/invitations/{invitation_id}", headers=self._auth(self.owner_token)
        )
        self.assertEqual(revoke_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()

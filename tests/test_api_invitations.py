#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/invitations/{token}、/api/invitations/{token}/accept 路由测试。"""

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


class TestInvitationRoutes(unittest.TestCase):
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

        owner_res = self.client.post("/api/auth/register", json={
            "email": "owner@example.com", "password": "password123", "name": "Owner",
        })
        self.owner_token = owner_res.json()["access_token"]
        tenant_res = self.client.post(
            "/api/tenants", json={"name": "工作区"}, headers={"Authorization": f"Bearer {self.owner_token}"}
        )
        self.tenant_id = tenant_res.json()["id"]

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("ragify.api.routers.tenants.send_invitation_email")
    def _create_invitation(self, mock_send, email: str = "invitee@example.com", role: str = "NORMAL") -> str:
        res = self.client.post(
            f"/api/tenants/{self.tenant_id}/invitations",
            json={"email": email, "role": role},
            headers={"Authorization": f"Bearer {self.owner_token}"},
        )
        invitation = self.invitation_manager.list_invitations(self.tenant_id)[-1]
        return invitation.token

    def test_get_invitation_by_token(self):
        token = self._create_invitation()
        res = self.client.get(f"/api/invitations/{token}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "invitee@example.com")
        self.assertEqual(res.json()["tenant_name"], "工作区")

    def test_get_invitation_missing_token_404(self):
        res = self.client.get("/api/invitations/not-a-real-token")
        self.assertEqual(res.status_code, 404)

    def test_accept_invitation_success(self):
        token = self._create_invitation(email="invitee@example.com")
        invitee_res = self.client.post("/api/auth/register", json={
            "email": "invitee@example.com", "password": "password123", "name": "Invitee",
        })
        invitee_token = invitee_res.json()["access_token"]

        res = self.client.post(
            f"/api/invitations/{token}/accept", headers={"Authorization": f"Bearer {invitee_token}"}
        )
        self.assertEqual(res.status_code, 200)

        invitee_id = self.user_manager.get_by_email("invitee@example.com").id
        membership = self.tenant_manager.get_membership(self.tenant_id, invitee_id)
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "NORMAL")

    def test_accept_invitation_wrong_account_rejected(self):
        token = self._create_invitation(email="invitee@example.com")
        wrong_res = self.client.post("/api/auth/register", json={
            "email": "someone-else@example.com", "password": "password123", "name": "Someone",
        })
        wrong_token = wrong_res.json()["access_token"]

        res = self.client.post(
            f"/api/invitations/{token}/accept", headers={"Authorization": f"Bearer {wrong_token}"}
        )
        self.assertEqual(res.status_code, 403)

    def test_accept_invitation_without_login_rejected(self):
        token = self._create_invitation()
        res = self.client.post(f"/api/invitations/{token}/accept")
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvitationManager 测试
验证邀请的建/查/撤销/接受，接受时的邮箱匹配、过期判断，以及并发接受时
的唯一约束竞态处理。
"""

import shutil
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine, get_session


class TestInvitationManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = InvitationManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)
        self.tenant = self.tenant_manager.create_tenant("工作区", "owner-1")

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_invitation(self):
        invitation = self.manager.create_invitation(self.tenant.id, "New@Example.com", "NORMAL", "owner-1")
        self.assertTrue(invitation.id)
        self.assertEqual(invitation.email, "new@example.com")
        self.assertEqual(invitation.status, "pending")
        self.assertTrue(invitation.token)

    def test_create_invitation_empty_email_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create_invitation(self.tenant.id, "   ", "NORMAL", "owner-1")

    def test_list_invitations(self):
        self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        self.manager.create_invitation(self.tenant.id, "b@example.com", "EDITOR", "owner-1")
        invitations = self.manager.list_invitations(self.tenant.id)
        self.assertEqual(len(invitations), 2)

    def test_get_by_token(self):
        invitation = self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        fetched = self.manager.get_by_token(invitation.token)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.email, "a@example.com")

    def test_get_by_token_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_token("not-a-real-token"))

    def test_revoke_invitation(self):
        invitation = self.manager.create_invitation(self.tenant.id, "a@example.com", "NORMAL", "owner-1")
        self.manager.revoke_invitation(self.tenant.id, invitation.id)
        fetched = self.manager.get_by_token(invitation.token)
        self.assertEqual(fetched.status, "revoked")

    def test_revoke_missing_invitation_raises(self):
        with self.assertRaises(ValueError):
            self.manager.revoke_invitation(self.tenant.id, "does-not-exist")

    def test_accept_invitation_creates_membership(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "EDITOR", "owner-1")
        self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

        membership = self.tenant_manager.get_membership(self.tenant.id, "user-2")
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "EDITOR")

        fetched = self.manager.get_by_token(invitation.token)
        self.assertEqual(fetched.status, "accepted")

    def test_accept_invitation_wrong_email_raises_permission_error(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        with self.assertRaises(PermissionError):
            self.manager.accept_invitation(invitation.token, "user-2", "someone-else@example.com")

    def test_accept_invitation_missing_token_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.accept_invitation("not-a-real-token", "user-2", "invitee@example.com")

    def test_accept_invitation_already_accepted_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-3", "invitee@example.com")

    def test_accept_invitation_revoked_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        self.manager.revoke_invitation(self.tenant.id, invitation.id)
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def test_accept_invitation_expired_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        with self._session() as session:
            from ragify.db.models import TenantInvitationRow
            row = session.get(TenantInvitationRow, invitation.id)
            row.expires_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            session.commit()

        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def test_accept_invitation_already_a_member_raises(self):
        invitation = self.manager.create_invitation(self.tenant.id, "owner-1@example.com", "NORMAL", "owner-1")
        # owner-1 已经是这个工作区的 OWNER 了（create_tenant 时建的）
        with self.assertRaises(ValueError):
            self.manager.accept_invitation(invitation.token, "owner-1", "owner-1@example.com")

    def test_accept_invitation_race_condition_raises_value_error(self):
        """跟 test_kb_manager.py/test_user_manager.py 的竞态测试同一个模式：
        两次并发 accept 同一个邀请（或者两个不同邀请但目标是同一个
        (tenant_id, user_id)）会撞上 TenantAccountJoinRow 的唯一约束，
        必须走 except IntegrityError -> ValueError 翻译，而不是让
        sqlalchemy.exc.IntegrityError 未处理地抛出去。"""
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        tenant_id = self.tenant.id
        original_commit = Session.commit
        state = {"injected": False}

        def racing_commit(session_self, *args, **kwargs):
            if not state["injected"]:
                state["injected"] = True
                other_session = get_session(self.db_url)
                try:
                    other_session.add(TenantAccountJoinRow(
                        id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id="user-2",
                        role="NORMAL", created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    original_commit(other_session)
                finally:
                    other_session.close()
            return original_commit(session_self, *args, **kwargs)

        with patch.object(Session, "commit", racing_commit):
            with self.assertRaises(ValueError):
                self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")

    def _session(self):
        return self.manager._session()


if __name__ == "__main__":
    unittest.main()

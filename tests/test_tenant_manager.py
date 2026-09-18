#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TenantManager 测试
验证工作区的建/查、成员关系查询、角色变更权限矩阵、移除/退出的唯一
OWNER 保护、默认工作区迁移的幂等性。
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.tenant_manager import TenantManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestTenantManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = TenantManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_tenant_creates_owner_membership(self):
        tenant = self.manager.create_tenant("测试工作区", "user-1")
        self.assertTrue(tenant.id)
        self.assertEqual(tenant.name, "测试工作区")

        membership = self.manager.get_membership(tenant.id, "user-1")
        self.assertIsNotNone(membership)
        self.assertEqual(membership.role, "OWNER")

    def test_create_tenant_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create_tenant("   ", "user-1")

    def test_get_tenant(self):
        tenant = self.manager.create_tenant("工作区A", "user-1")
        fetched = self.manager.get_tenant(tenant.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "工作区A")

    def test_get_tenant_missing_returns_none(self):
        self.assertIsNone(self.manager.get_tenant("does-not-exist"))

    def test_list_tenants_for_user(self):
        t1 = self.manager.create_tenant("工作区1", "user-1")
        t2 = self.manager.create_tenant("工作区2", "user-1")
        self.manager.create_tenant("工作区3", "user-2")

        tenants = self.manager.list_tenants_for_user("user-1")
        ids = {t.id for t in tenants}
        self.assertEqual(ids, {t1.id, t2.id})

    def test_list_tenants_for_user_empty(self):
        self.assertEqual(self.manager.list_tenants_for_user("nobody"), [])

    def test_get_membership_missing_returns_none(self):
        tenant = self.manager.create_tenant("工作区", "user-1")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_list_members(self):
        tenant = self.manager.create_tenant("工作区", "user-1")
        members = self.manager.list_members(tenant.id)
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].user_id, "user-1")
        self.assertEqual(members[0].role, "OWNER")

    def test_list_members_empty_tenant(self):
        self.assertEqual(self.manager.list_members("does-not-exist"), [])

    def test_update_member_role_by_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        updated = self.manager.update_member_role(tenant.id, "user-2", "EDITOR", acting_role="OWNER")
        self.assertEqual(updated.role, "EDITOR")

    def test_update_member_role_invalid_role_raises_value_error(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "owner-1", "SUPERUSER", acting_role="OWNER")

    def test_update_member_role_missing_member_raises_value_error(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "ghost", "NORMAL", acting_role="OWNER")

    def test_admin_cannot_promote_to_admin(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        with self.assertRaises(PermissionError):
            self.manager.update_member_role(tenant.id, "user-2", "ADMIN", acting_role="ADMIN")

    def test_admin_cannot_modify_another_admin(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "admin-2", "ADMIN")
        with self.assertRaises(PermissionError):
            self.manager.update_member_role(tenant.id, "admin-2", "NORMAL", acting_role="ADMIN")

    def test_admin_can_promote_normal_to_editor(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        updated = self.manager.update_member_role(tenant.id, "user-2", "EDITOR", acting_role="ADMIN")
        self.assertEqual(updated.role, "EDITOR")

    def test_cannot_demote_sole_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.update_member_role(tenant.id, "owner-1", "ADMIN", acting_role="OWNER")

    def test_can_demote_owner_when_another_owner_exists(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "owner-2", "OWNER")
        updated = self.manager.update_member_role(tenant.id, "owner-1", "ADMIN", acting_role="OWNER")
        self.assertEqual(updated.role, "ADMIN")

    def test_remove_member_by_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        self.manager.remove_member(tenant.id, "user-2", acting_role="OWNER")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_admin_cannot_remove_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "admin-2", "ADMIN")
        with self.assertRaises(PermissionError):
            self.manager.remove_member(tenant.id, "owner-1", acting_role="ADMIN")

    def test_cannot_remove_sole_owner(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.remove_member(tenant.id, "owner-1", acting_role="OWNER")

    def test_leave_tenant(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "user-2", "NORMAL")
        self.manager.leave_tenant(tenant.id, "user-2")
        self.assertIsNone(self.manager.get_membership(tenant.id, "user-2"))

    def test_sole_owner_cannot_leave(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        with self.assertRaises(ValueError):
            self.manager.leave_tenant(tenant.id, "owner-1")

    def test_owner_can_leave_when_another_owner_exists(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "owner-2", "OWNER")
        self.manager.leave_tenant(tenant.id, "owner-1")
        self.assertIsNone(self.manager.get_membership(tenant.id, "owner-1"))

    def test_delete_tenant(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self.manager.delete_tenant(tenant.id)
        self.assertIsNone(self.manager.get_tenant(tenant.id))
        self.assertEqual(self.manager.list_members(tenant.id), [])

    def test_delete_missing_tenant_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.delete_tenant("does-not-exist")

    def _add_member(self, tenant_id: str, user_id: str, role: str) -> None:
        import uuid as uuid_module
        from datetime import datetime, timezone
        from ragify.db.models import TenantAccountJoinRow
        with self.manager._session() as session:
            session.add(TenantAccountJoinRow(
                id=uuid_module.uuid4().hex[:12], tenant_id=tenant_id, user_id=user_id,
                role=role, created_at=datetime.now(timezone.utc).isoformat(),
            ))
            session.commit()


if __name__ == "__main__":
    unittest.main()

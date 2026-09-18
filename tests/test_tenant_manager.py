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
import threading
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

    def test_concurrent_remove_last_two_owners_is_serialized(self):
        """证明 remove_member 的唯一 OWNER 保护在真并发下不会被绕过。

        这是 code review 发现的竞态：update_member_role/remove_member/
        leave_tenant 都是"数一下还有几个 OWNER，再决定能不能改/删"的
        check-then-act。两个线程同时对同一个只有两个 OWNER 的工作区调用
        remove_member，各自移除其中一个 OWNER，在没有任何人工延迟的情况下，
        两个线程都能在对方提交前读到 owner_count == 2、都通过校验、都提交
        成功——工作区就会同时失去所有 OWNER（多次实测命中率约 40%）。

        跟 test_user_manager.py 的 test_create_race_condition_raises_value_error
        不同，那个测试用 monkeypatch Session.commit 在单线程里注入一次"插队"
        写入就足够复现问题，因为竞争的是数据库的 UNIQUE 约束，不涉及锁。但这里
        修复用的是进程内 threading.Lock 按 tenant_id 序列化，monkeypatch-commit
        技术在单线程执行流里不会有第二个线程去竞争同一把锁，没法证明锁本身生效。
        所以这里改用真正的 threading.Thread + threading.Barrier(2)：两个线程都
        卡在 barrier 上，同时被放行去调用 remove_member，逼出与 reviewer 复现
        时完全一样的时间窗口。加了 _get_tenant_lock 之后，两个线程会被强制
        序列化——先拿到锁的线程读到 owner_count == 2、成功移除；后拿到锁的
        线程读到的是移除之后的最新状态（owner_count == 1），必须抛出
        ValueError。断言：恰好一个线程成功、恰好一个线程抛出 ValueError，且
        工作区最终仍有且只有一个 OWNER——不会出现零 OWNER 的不可恢复状态。
        """
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self._add_member(tenant.id, "owner-2", "OWNER")

        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def remove(user_id: str, key: str) -> None:
            barrier.wait()
            try:
                self.manager.remove_member(tenant.id, user_id, acting_role="OWNER")
                results[key] = "ok"
            except ValueError as exc:
                results[key] = exc

        t1 = threading.Thread(target=remove, args=("owner-1", "t1"))
        t2 = threading.Thread(target=remove, args=("owner-2", "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        outcomes = list(results.values())
        successes = [o for o in outcomes if o == "ok"]
        failures = [o for o in outcomes if isinstance(o, ValueError)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)

        remaining_members = self.manager.list_members(tenant.id)
        remaining_owners = [m for m in remaining_members if m.role == "OWNER"]
        self.assertEqual(len(remaining_owners), 1)

    def test_delete_tenant(self):
        tenant = self.manager.create_tenant("工作区", "owner-1")
        self.manager.delete_tenant(tenant.id)
        self.assertIsNone(self.manager.get_tenant(tenant.id))
        self.assertEqual(self.manager.list_members(tenant.id), [])

    def test_delete_missing_tenant_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.manager.delete_tenant("does-not-exist")

    def test_migrate_default_tenant_creates_default_workspace(self):
        self._insert_user("user-1", "2024-01-01T00:00:00+00:00")
        self._insert_user("user-2", "2024-01-02T00:00:00+00:00")

        migrated = self.manager.migrate_default_tenant_if_needed()

        self.assertTrue(migrated)
        tenants = self.manager.list_tenants_for_user("user-1")
        self.assertEqual(len(tenants), 1)
        self.assertEqual(tenants[0].name, "默认工作区")

        owner_membership = self.manager.get_membership(tenants[0].id, "user-1")
        self.assertEqual(owner_membership.role, "OWNER")
        admin_membership = self.manager.get_membership(tenants[0].id, "user-2")
        self.assertEqual(admin_membership.role, "ADMIN")

    def test_migrate_default_tenant_noop_when_tenant_exists(self):
        self._insert_user("user-1", "2024-01-01T00:00:00+00:00")
        self.manager.create_tenant("已有工作区", "user-1")

        migrated = self.manager.migrate_default_tenant_if_needed()

        self.assertFalse(migrated)
        self.assertEqual(len(self.manager.list_tenants_for_user("user-1")), 1)

    def test_migrate_default_tenant_noop_when_no_users(self):
        self.assertFalse(self.manager.migrate_default_tenant_if_needed())

    def _insert_user(self, user_id: str, created_at: str) -> None:
        from ragify.db.models import UserRow
        with self.manager._session() as session:
            session.add(UserRow(
                id=user_id, email=f"{user_id}@example.com", password_hash="x",
                name=user_id, created_at=created_at,
            ))
            session.commit()

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

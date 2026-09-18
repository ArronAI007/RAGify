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
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.invitation_manager import InvitationManager
from ragify.core.tenant_manager import TenantManager
from ragify.db.models import Base, TenantAccountJoinRow
from ragify.db.session import get_engine


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
        """两次并发写入同一个 (tenant_id, user_id) 会撞上 TenantAccountJoinRow
        的唯一约束，必须走 except IntegrityError -> ValueError 翻译，而不是让
        sqlalchemy.exc.IntegrityError 未处理地抛出去。

        注意：这里改用真实线程 + Barrier，而不是像之前那样 mock
        Session.commit 在它内部同步地插入一个"别的"竞争写入。原来那种 mock
        技巧假设写入只会发生在 session.commit() 调用的那一刻——这在旧的
        "先读 status 再直接赋值"实现下成立，但修复后 accept_invitation 会在
        commit 之前就用 session.execute(update(...)) 立即发出一条真正的写
        SQL（拿到 SQLite 的写锁，直到这个 session 自己 commit/rollback 才
        释放）。如果还是在同一个线程里、在这条 UPDATE 已经拿到锁但还没提交
        的窗口内，同步地开一个"别的" session 去 commit 一次写入，两者其实
        是死锁的（另一个 session 只能等锁，但锁的持有者要等它先返回才会去
        commit）——SQLite 的 busy_timeout（默认 5 秒）耗尽后会直接抛出
        OperationalError("database is locked")，而不是我们想验证的
        IntegrityError -> ValueError 路径。用真实的两个线程并发，两者是
        真正并行执行的，先完成的一方会正常提交并释放锁，另一方在
        busy_timeout 内重试后能拿到锁继续执行，这样才能如实复现"两次插入
        同一个 (tenant_id, user_id)，恰好一次撞上唯一约束"的场景。"""
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")
        tenant_id = self.tenant.id

        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def do_accept() -> None:
            barrier.wait()
            try:
                self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")
                results["accept"] = "ok"
            except ValueError as exc:
                results["accept"] = exc

        def do_direct_insert() -> None:
            barrier.wait()
            session = self.manager._session()
            try:
                session.add(TenantAccountJoinRow(
                    id=uuid.uuid4().hex[:12], tenant_id=tenant_id, user_id="user-2",
                    role="NORMAL", created_at=datetime.now(timezone.utc).isoformat(),
                ))
                session.commit()
                results["direct"] = "ok"
            except IntegrityError as exc:
                session.rollback()
                results["direct"] = exc
            finally:
                session.close()

        t1 = threading.Thread(target=do_accept)
        t2 = threading.Thread(target=do_direct_insert)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        outcomes = list(results.values())
        successes = [o for o in outcomes if o == "ok"]
        failures = [o for o in outcomes if isinstance(o, (ValueError, IntegrityError))]
        # 两边都想给 (tenant_id="user-2 所在的 tenant", user_id="user-2") 建
        # 一行成员关系，唯一约束保证只有一个能赢；哪个赢是并发调度决定的，
        # 但绝不能两个都成功。
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)

    def test_concurrent_accept_and_revoke_is_serialized(self):
        """证明 accept_invitation 和 revoke_invitation 并发操作同一个邀请时
        不会同时成功。用真实线程 + Barrier 逼出与 code review 复现时完全
        一样的竞态窗口：一个线程 accept，另一个线程 revoke，同时对同一个
        邀请发起调用。修复前两个调用都会成功（accept 建出成员关系,
        invitation.status 变成 accept 和 revoke 里最后提交的那个,
        是纯粹的巧合)——修复后两者必须恰好一个成功、一个抛 ValueError。"""
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")

        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def do_accept() -> None:
            barrier.wait()
            try:
                self.manager.accept_invitation(invitation.token, "user-2", "invitee@example.com")
                results["accept"] = "ok"
            except ValueError as exc:
                results["accept"] = exc

        def do_revoke() -> None:
            barrier.wait()
            try:
                self.manager.revoke_invitation(self.tenant.id, invitation.id)
                results["revoke"] = "ok"
            except ValueError as exc:
                results["revoke"] = exc

        t1 = threading.Thread(target=do_accept)
        t2 = threading.Thread(target=do_revoke)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        outcomes = list(results.values())
        successes = [o for o in outcomes if o == "ok"]
        failures = [o for o in outcomes if isinstance(o, ValueError)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)

        # 无论谁赢，都不能出现"成员关系建立了，但状态还认为邀请是 pending"
        # 这种矛盾状态——最终状态必须是 accepted 或 revoked 二选一，跟是否
        # 真的建出了成员关系一致。
        final = self.manager.get_by_token(invitation.token)
        membership = self.tenant_manager.get_membership(self.tenant.id, "user-2")
        if final.status == "accepted":
            self.assertIsNotNone(membership)
        else:
            self.assertEqual(final.status, "revoked")
            self.assertIsNone(membership)

    def test_concurrent_accept_by_two_different_users_only_one_succeeds(self):
        """证明同一个 token 不能被两个不同的 user_id 并发 accept 两次。
        (tenant_id, user_id) 唯一约束在这个场景下根本不会冲突（两个 user_id
        不一样），所以原来的 except IntegrityError 兜底完全防不住这个——
        必须靠 accept_invitation 内部对 status 的原子条件更新来堵这个口子。"""
        invitation = self.manager.create_invitation(self.tenant.id, "invitee@example.com", "NORMAL", "owner-1")

        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def accept_as(user_id: str, key: str) -> None:
            barrier.wait()
            try:
                self.manager.accept_invitation(invitation.token, user_id, "invitee@example.com")
                results[key] = "ok"
            except ValueError as exc:
                results[key] = exc

        t1 = threading.Thread(target=accept_as, args=("user-a", "a"))
        t2 = threading.Thread(target=accept_as, args=("user-b", "b"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        outcomes = list(results.values())
        successes = [o for o in outcomes if o == "ok"]
        failures = [o for o in outcomes if isinstance(o, ValueError)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)

        members = self.tenant_manager.list_members(self.tenant.id)
        member_ids = {m.user_id for m in members}
        # 只有 owner-1（建工作区时自带）加上恰好一个 accept 成功的人，不能
        # 两个都进来。
        self.assertTrue(("user-a" in member_ids) != ("user-b" in member_ids))

    def _session(self):
        return self.manager._session()


if __name__ == "__main__":
    unittest.main()

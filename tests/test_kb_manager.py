#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KBManager 测试（DB 驱动版，Phase 4 加了 tenant_id 归属）
验证 KB 的增删查、工作区内名称去重、跨工作区隔离，以及从 kbs.json / 旧版
扁平索引迁移进数据库的逻辑。
"""

import json
import shutil
import sys
import tempfile
import threading
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.kb_manager import KBManager
from ragify.db.models import Base, KnowledgeBaseRow
from ragify.db.session import get_engine, get_session

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


class TestKBManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        self.vectorstore_dir = Path(self.tmp_dir) / "vectorstore"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = KBManager(database_url=self.db_url, vectorstore_dir=self.vectorstore_dir)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_get(self):
        kb = self.manager.create("测试知识库", "描述", TENANT_A)
        self.assertTrue(kb.id)
        self.assertEqual(kb.tenant_id, TENANT_A)
        fetched = self.manager.get(kb.id, TENANT_A)
        self.assertEqual(fetched.name, "测试知识库")
        self.assertEqual(fetched.description, "描述")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("   ", "", TENANT_A)

    def test_create_duplicate_name_within_same_tenant_raises(self):
        self.manager.create("重复名称", "", TENANT_A)
        with self.assertRaises(ValueError):
            self.manager.create("重复名称", "", TENANT_A)

    def test_create_same_name_in_different_tenants_both_succeed(self):
        kb_a = self.manager.create("同名知识库", "", TENANT_A)
        kb_b = self.manager.create("同名知识库", "", TENANT_B)
        self.assertNotEqual(kb_a.id, kb_b.id)
        self.assertEqual(self.manager.get(kb_a.id, TENANT_A).name, "同名知识库")
        self.assertEqual(self.manager.get(kb_b.id, TENANT_B).name, "同名知识库")

    def test_create_race_condition_raises_value_error(self):
        """Exercises the `except IntegrityError` branch in create(), not the
        in-Python pre-check (that path is already covered by
        test_create_duplicate_name_within_same_tenant_raises).

        Same technique as before Phase 4: patch Session.commit so that the
        *first* time create() calls it, we commit a colliding
        (tenant_id, name) row through a *separate* session first, simulating
        another process's create() call finishing in that exact window.
        """
        name = "并发冲突名称"
        original_commit = Session.commit
        state = {"injected": False}

        def racing_commit(session_self, *args, **kwargs):
            if not state["injected"]:
                state["injected"] = True
                other_session = get_session(self.db_url)
                try:
                    other_session.add(KnowledgeBaseRow(
                        id=uuid.uuid4().hex[:12], tenant_id=TENANT_A, name=name, description="",
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    original_commit(other_session)
                finally:
                    other_session.close()
            return original_commit(session_self, *args, **kwargs)

        with patch.object(Session, "commit", racing_commit):
            with self.assertRaises(ValueError):
                self.manager.create(name, "", TENANT_A)

    def test_concurrent_create_different_case_same_name_only_one_succeeds(self):
        """证明大小写不同但视为同名的并发创建不会绕过工作区内唯一约束。
        两个线程同时对同一个 tenant 调用 create("KB", ...) 和 create("kb", ...)。
        数据库的 UniqueConstraint(tenant_id, name) 本身是大小写敏感的，不会
        拦住这一对；靠 create() 内部按 tenant_id 加的锁把两次调用强制序列化，
        后执行的那次在锁内重新读到已经存在的名字，必须失败。"""
        barrier = threading.Barrier(2)
        results: dict[str, object] = {}

        def create_variant(variant_name: str, key: str) -> None:
            barrier.wait()
            try:
                kb = self.manager.create(variant_name, "", TENANT_A)
                results[key] = kb
            except ValueError as exc:
                results[key] = exc

        t1 = threading.Thread(target=create_variant, args=("KB", "upper"))
        t2 = threading.Thread(target=create_variant, args=("kb", "lower"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        outcomes = list(results.values())
        successes = [o for o in outcomes if not isinstance(o, Exception)]
        failures = [o for o in outcomes if isinstance(o, ValueError)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)

        remaining = self.manager.list_all(TENANT_A)
        self.assertEqual(len(remaining), 1)

    def test_list_all_empty(self):
        self.assertEqual(self.manager.list_all(TENANT_A), [])

    def test_list_all_only_returns_own_tenant(self):
        self.manager.create("A的知识库", "", TENANT_A)
        self.manager.create("B的知识库", "", TENANT_B)

        kbs_a = self.manager.list_all(TENANT_A)
        self.assertEqual(len(kbs_a), 1)
        self.assertEqual(kbs_a[0].name, "A的知识库")

        kbs_b = self.manager.list_all(TENANT_B)
        self.assertEqual(len(kbs_b), 1)
        self.assertEqual(kbs_b[0].name, "B的知识库")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.manager.get("does-not-exist", TENANT_A))

    def test_get_with_wrong_tenant_returns_none(self):
        """跨租户隔离的核心断言：即使 kb_id 是真实存在的（比如从别的工作区
        的旧链接、日志里泄露出来），用别的 tenant_id 去 get 也必须表现得
        跟"这个 kb_id 根本不存在"完全一样——不能因为 kb_id 本身合法就泄露
        任何信息（哪怕只是 404 vs 拿到数据的区别）。"""
        kb = self.manager.create("A的知识库", "", TENANT_A)
        self.assertIsNone(self.manager.get(kb.id, TENANT_B))
        self.assertIsNotNone(self.manager.get(kb.id, TENANT_A))

    def test_delete_removes_kb_and_directory(self):
        kb = self.manager.create("待删除", "", TENANT_A)
        # 注意：get_persist_dir 本任务保持不变（单参数、非租户嵌套路径），
        # 而 create/delete 已改为使用租户嵌套路径
        # (vectorstore_dir/tenant_id/kb_id)，Task 3 才会让 get_persist_dir
        # 跟上这个新布局。这里直接拼真实路径来验证目录的创建与删除。
        kb_dir = self.vectorstore_dir / TENANT_A / kb.id
        self.assertTrue(kb_dir.exists())

        ok = self.manager.delete(kb.id, TENANT_A)

        self.assertTrue(ok)
        self.assertIsNone(self.manager.get(kb.id, TENANT_A))
        self.assertFalse(kb_dir.exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.manager.delete("does-not-exist", TENANT_A))

    def test_delete_with_wrong_tenant_returns_false_and_does_not_delete(self):
        """同 test_get_with_wrong_tenant_returns_none 的隔离要求：用别的
        tenant_id 删不掉别人的知识库，行为跟"不存在"一样，且知识库本身
        必须还在。"""
        kb = self.manager.create("A的知识库", "", TENANT_A)
        ok = self.manager.delete(kb.id, TENANT_B)
        self.assertFalse(ok)
        self.assertIsNotNone(self.manager.get(kb.id, TENANT_A))

    def test_migrate_json_if_needed_imports_existing_file(self):
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        kbs_file = self.vectorstore_dir / "kbs.json"
        kbs_file.write_text(json.dumps({"kbs": [
            {"id": "legacy1", "name": "旧知识库", "description": "", "created_at": "2024-01-01T00:00:00"}
        ]}), encoding="utf-8")

        migrated = self.manager.migrate_json_if_needed()

        self.assertTrue(migrated)
        # 迁移进来的行还没有 tenant_id（那是 migrate_tenant_id_if_needed 的
        # 职责，Task 3 才会加），这里用底层查询确认行本身确实进了库。
        with self.manager._session() as session:
            row = session.get(KnowledgeBaseRow, "legacy1")
            self.assertIsNotNone(row)
            self.assertEqual(row.name, "旧知识库")
            self.assertIsNone(row.tenant_id)
        self.assertFalse(kbs_file.exists())
        self.assertTrue((self.vectorstore_dir / "kbs.json.migrated").exists())

    def test_migrate_json_if_needed_noop_when_db_has_rows(self):
        self.manager.create("已有数据", "", TENANT_A)
        self.assertFalse(self.manager.migrate_json_if_needed())

    def test_migrate_json_if_needed_noop_when_nothing_to_migrate(self):
        self.assertFalse(self.manager.migrate_json_if_needed())


if __name__ == "__main__":
    unittest.main()

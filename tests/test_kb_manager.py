#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KBManager 测试（DB 驱动版）
验证 KB 的增删查、名称去重，以及从 kbs.json / 旧版扁平索引迁移进数据库的逻辑。
"""

import json
import shutil
import sys
import tempfile
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
        kb = self.manager.create("测试知识库", "描述")
        self.assertTrue(kb.id)
        fetched = self.manager.get(kb.id)
        self.assertEqual(fetched.name, "测试知识库")
        self.assertEqual(fetched.description, "描述")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("   ")

    def test_create_duplicate_name_raises(self):
        self.manager.create("重复名称")
        with self.assertRaises(ValueError):
            self.manager.create("重复名称")

    def test_create_race_condition_raises_value_error(self):
        """Exercises the `except IntegrityError` branch in create(), not the
        in-Python pre-check (that path is already covered by
        test_create_duplicate_name_raises).

        A true concurrent-thread interleaving isn't practical to assert
        deterministically here, so instead we force the same interleaving
        deterministically: we patch Session.commit so that the *first* time
        create() calls it (i.e. right after its own pre-check already ran
        and found no duplicate), we commit a colliding row through a
        *separate* session first -- simulating another process's create()
        call finishing in that exact window. create()'s own subsequent
        commit then hits the DB's UNIQUE constraint and must go through the
        except IntegrityError -> ValueError translation, since its pre-check
        already ran before the collision existed. If the try/except
        IntegrityError block were removed, this test would fail with an
        unhandled sqlalchemy.exc.IntegrityError instead of ValueError.
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
                        id=uuid.uuid4().hex[:12], name=name, description="",
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    # Call the unpatched commit directly so this doesn't
                    # recurse into racing_commit again.
                    original_commit(other_session)
                finally:
                    other_session.close()
            return original_commit(session_self, *args, **kwargs)

        with patch.object(Session, "commit", racing_commit):
            with self.assertRaises(ValueError):
                self.manager.create(name)

    def test_list_all_empty(self):
        self.assertEqual(self.manager.list_all(), [])

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.manager.get("does-not-exist"))

    def test_delete_removes_kb_and_directory(self):
        kb = self.manager.create("待删除")
        kb_dir = Path(self.manager.get_persist_dir(kb.id))
        self.assertTrue(kb_dir.exists())

        ok = self.manager.delete(kb.id)

        self.assertTrue(ok)
        self.assertIsNone(self.manager.get(kb.id))
        self.assertFalse(kb_dir.exists())

    def test_delete_missing_returns_false(self):
        self.assertFalse(self.manager.delete("does-not-exist"))

    def test_migrate_json_if_needed_imports_existing_file(self):
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        kbs_file = self.vectorstore_dir / "kbs.json"
        kbs_file.write_text(json.dumps({"kbs": [
            {"id": "legacy1", "name": "旧知识库", "description": "", "created_at": "2024-01-01T00:00:00"}
        ]}), encoding="utf-8")

        migrated = self.manager.migrate_json_if_needed()

        self.assertTrue(migrated)
        kb = self.manager.get("legacy1")
        self.assertIsNotNone(kb)
        self.assertEqual(kb.name, "旧知识库")
        self.assertFalse(kbs_file.exists())
        self.assertTrue((self.vectorstore_dir / "kbs.json.migrated").exists())

    def test_migrate_json_if_needed_noop_when_db_has_rows(self):
        self.manager.create("已有数据")
        self.assertFalse(self.manager.migrate_json_if_needed())

    def test_migrate_json_if_needed_noop_when_nothing_to_migrate(self):
        self.assertFalse(self.manager.migrate_json_if_needed())


if __name__ == "__main__":
    unittest.main()

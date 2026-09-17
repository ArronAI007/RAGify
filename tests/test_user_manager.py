#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UserManager 测试
验证用户的增查、邮箱去重（含大小写归一化）、密码校验。
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = UserManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_create_and_get_by_email(self):
        user = self.manager.create("Alice@Example.com", "password123", "Alice")
        self.assertTrue(user.id)
        self.assertEqual(user.email, "alice@example.com")

        fetched = self.manager.get_by_email("alice@example.com")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "Alice")

    def test_get_by_email_is_case_insensitive(self):
        self.manager.create("Bob@Example.com", "password123", "Bob")
        fetched = self.manager.get_by_email("BOB@EXAMPLE.COM")
        self.assertIsNotNone(fetched)

    def test_create_duplicate_email_raises(self):
        self.manager.create("dup@example.com", "password123", "First")
        with self.assertRaises(ValueError):
            self.manager.create("dup@example.com", "password456", "Second")

    def test_create_duplicate_email_different_case_raises(self):
        self.manager.create("case@example.com", "password123", "First")
        with self.assertRaises(ValueError):
            self.manager.create("Case@Example.com", "password456", "Second")

    def test_create_short_password_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("short@example.com", "1234567", "Short")

    def test_create_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self.manager.create("noname@example.com", "password123", "   ")

    def test_get_by_email_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_email("nobody@example.com"))

    def test_get_by_id_missing_returns_none(self):
        self.assertIsNone(self.manager.get_by_id("does-not-exist"))

    def test_get_by_id(self):
        user = self.manager.create("byid@example.com", "password123", "ById")
        fetched = self.manager.get_by_id(user.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.email, "byid@example.com")

    def test_verify_credentials_correct(self):
        self.manager.create("verify@example.com", "correct-password", "Verify")
        user = self.manager.verify_credentials("verify@example.com", "correct-password")
        self.assertIsNotNone(user)
        self.assertEqual(user.email, "verify@example.com")

    def test_verify_credentials_wrong_password_returns_none(self):
        self.manager.create("verify2@example.com", "correct-password", "Verify2")
        user = self.manager.verify_credentials("verify2@example.com", "wrong-password")
        self.assertIsNone(user)

    def test_verify_credentials_missing_email_returns_none(self):
        user = self.manager.verify_credentials("nobody2@example.com", "whatever123")
        self.assertIsNone(user)


if __name__ == "__main__":
    unittest.main()

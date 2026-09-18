#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""/api/auth/register、/api/auth/login、/api/auth/me 路由测试。"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.dependencies import get_user_manager
from ragify.api.main import app
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine


class TestAuthRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.manager = UserManager(database_url=self.db_url)
        app.dependency_overrides[get_user_manager] = lambda: self.manager
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_register_returns_token_and_user(self):
        res = self.client.post("/api/auth/register", json={
            "email": "new@example.com", "password": "password123", "name": "New User",
        })
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["email"], "new@example.com")
        self.assertEqual(body["user"]["name"], "New User")

    def test_register_duplicate_email_rejected(self):
        self.client.post("/api/auth/register", json={
            "email": "dup@example.com", "password": "password123", "name": "First",
        })
        res = self.client.post("/api/auth/register", json={
            "email": "dup@example.com", "password": "password456", "name": "Second",
        })
        self.assertEqual(res.status_code, 400)

    def test_register_short_password_rejected(self):
        res = self.client.post("/api/auth/register", json={
            "email": "shortpw@example.com", "password": "short", "name": "Short",
        })
        self.assertEqual(res.status_code, 400)

    def test_register_invalid_email_rejected(self):
        res = self.client.post("/api/auth/register", json={
            "email": "not-an-email", "password": "password123", "name": "Bad Email",
        })
        self.assertEqual(res.status_code, 422)  # Pydantic EmailStr 校验，路由函数都还没进

    def test_login_correct_credentials(self):
        self.client.post("/api/auth/register", json={
            "email": "login@example.com", "password": "password123", "name": "Login User",
        })
        res = self.client.post("/api/auth/login", json={
            "email": "login@example.com", "password": "password123",
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn("access_token", res.json())

    def test_login_wrong_password_rejected(self):
        self.client.post("/api/auth/register", json={
            "email": "login2@example.com", "password": "password123", "name": "Login User 2",
        })
        res = self.client.post("/api/auth/login", json={
            "email": "login2@example.com", "password": "wrong-password",
        })
        self.assertEqual(res.status_code, 401)

    def test_login_unknown_email_rejected(self):
        res = self.client.post("/api/auth/login", json={
            "email": "ghost@example.com", "password": "whatever123",
        })
        self.assertEqual(res.status_code, 401)

    def test_me_with_valid_token(self):
        register_res = self.client.post("/api/auth/register", json={
            "email": "me@example.com", "password": "password123", "name": "Me User",
        })
        token = register_res.json()["access_token"]
        res = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["email"], "me@example.com")

    def test_me_without_token_rejected(self):
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_me_with_garbage_token_rejected(self):
        res = self.client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main()

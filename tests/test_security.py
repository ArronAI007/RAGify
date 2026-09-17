#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""密码哈希/校验 + JWT 编码/解码的单测。纯函数测试，不碰数据库。"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jwt as pyjwt

from ragify.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

TEST_SECRET = "test-secret-only-for-unit-tests"


class TestPasswordHashing(unittest.TestCase):
    def test_hash_is_not_the_plaintext(self):
        h = hash_password("correct horse battery staple")
        self.assertNotEqual(h, "correct horse battery staple")

    def test_same_password_hashes_differently_each_time(self):
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        self.assertNotEqual(h1, h2)  # bcrypt 每次用不同的随机 salt

    def test_verify_password_correct(self):
        h = hash_password("my-password")
        self.assertTrue(verify_password("my-password", h))

    def test_verify_password_wrong(self):
        h = hash_password("my-password")
        self.assertFalse(verify_password("wrong-password", h))

    def test_hash_password_rejects_over_72_bytes(self):
        with self.assertRaises(ValueError):
            hash_password("a" * 73)

    def test_verify_password_rejects_over_72_byte_candidate(self):
        h = hash_password("a" * 72)
        with self.assertRaises(ValueError):
            verify_password("a" * 73, h)


class TestJWT(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        token = create_access_token("user-123", "a@b.com", secret=TEST_SECRET)
        payload = decode_access_token(token, secret=TEST_SECRET)
        self.assertEqual(payload["sub"], "user-123")
        self.assertEqual(payload["email"], "a@b.com")

    def test_decode_with_wrong_secret_raises(self):
        token = create_access_token("user-123", "a@b.com", secret=TEST_SECRET)
        with self.assertRaises(pyjwt.PyJWTError):
            decode_access_token(token, secret="a-different-secret")

    def test_decode_expired_token_raises(self):
        # 手工构造一个已经过期的 token，绕开 create_access_token 固定 7 天
        # 有效期的默认值，直接测过期校验本身。
        expired_payload = {
            "sub": "user-123",
            "email": "a@b.com",
            "iat": datetime.now(timezone.utc) - timedelta(days=8),
            "exp": datetime.now(timezone.utc) - timedelta(days=1),
        }
        expired_token = pyjwt.encode(expired_payload, TEST_SECRET, algorithm="HS256")
        with self.assertRaises(pyjwt.ExpiredSignatureError):
            decode_access_token(expired_token, secret=TEST_SECRET)

    def test_decode_garbage_token_raises(self):
        with self.assertRaises(pyjwt.PyJWTError):
            decode_access_token("not-a-real-token", secret=TEST_SECRET)

    def test_decode_rejects_alg_none_token(self):
        import base64
        import json

        def _b64url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        payload = _b64url(json.dumps({"sub": "attacker", "email": "x@y.com"}).encode())
        forged_token = f"{header}.{payload}."
        with self.assertRaises(pyjwt.PyJWTError):
            decode_access_token(forged_token, secret=TEST_SECRET)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI /api/health 路由测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestHealthRoute(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_200_with_expected_shape(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("status", body)
        self.assertIn("version", body)
        self.assertIn("llm_provider", body)
        self.assertIn("vectorstore_type", body)


if __name__ == "__main__":
    unittest.main()

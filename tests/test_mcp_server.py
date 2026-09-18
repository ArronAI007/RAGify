#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP server 的租户识别测试。
验证 RAGIFY_MCP_TOKEN 缺失/无效/用户不存在/用户无工作区几种启动失败场景，
以及正常场景下解出的 tenant_id 确实被传给 KBManager 的调用。
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.core.security import create_access_token
from ragify.core.tenant_manager import TenantManager
from ragify.core.user_manager import UserManager
from ragify.db.models import Base
from ragify.db.session import get_engine
from ragify.mcp_server.server import _call_tool, _list_resources, _resolve_mcp_tenant_id

TEST_SECRET = "test-secret-only-for-unit-tests"


class TestResolveMcpTenantId(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_url = f"sqlite:///{self.tmp_dir}/test.db"
        Base.metadata.create_all(bind=get_engine(self.db_url))
        self.user_manager = UserManager(database_url=self.db_url)
        self.tenant_manager = TenantManager(database_url=self.db_url)

    def tearDown(self):
        get_engine.cache_clear()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_token_raises(self):
        with self.assertRaises(RuntimeError):
            _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_invalid_token_raises(self):
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": "not-a-real-token"}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_user_not_found_raises(self):
        token = create_access_token("ghost-user-id", "ghost@example.com", secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_user_with_no_tenants_raises(self):
        user = self.user_manager.create("solo@example.com", "password123", "Solo")
        token = create_access_token(user.id, user.email, secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            with self.assertRaises(RuntimeError):
                _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)

    def test_success_returns_first_tenant(self):
        user = self.user_manager.create("owner@example.com", "password123", "Owner")
        tenant = self.tenant_manager.create_tenant("工作区", user.id)
        token = create_access_token(user.id, user.email, secret=TEST_SECRET)
        with patch.dict(os.environ, {"RAGIFY_MCP_TOKEN": token}, clear=True):
            tenant_id = _resolve_mcp_tenant_id(self.user_manager, self.tenant_manager, secret=TEST_SECRET)
        self.assertEqual(tenant_id, tenant.id)


class TestMcpToolsTenantScoping(unittest.TestCase):
    @patch("ragify.mcp_server.server.KBManager")
    def test_list_resources_passes_tenant_id(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        mock_manager.list_all.return_value = []
        _list_resources("tenant-x")
        mock_manager.list_all.assert_called_once_with("tenant-x")

    @patch("ragify.mcp_server.server.KBManager")
    def test_call_tool_list_kbs_passes_tenant_id(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        mock_manager.list_all.return_value = []
        _call_tool("ragify_list_kbs", {}, "tenant-y")
        mock_manager.list_all.assert_called_once_with("tenant-y")


if __name__ == "__main__":
    unittest.main()

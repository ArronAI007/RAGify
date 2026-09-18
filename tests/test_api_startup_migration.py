#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 FastAPI 启动事件会依次调用 KBManager.migrate_json_if_needed()、
TenantManager.migrate_default_tenant_if_needed()、
KBManager.migrate_tenant_id_if_needed()、
KBManager.migrate_vectorstore_layout_if_needed()。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestStartupMigration(unittest.TestCase):
    @patch("ragify.api.main.TenantManager")
    @patch("ragify.api.main.KBManager")
    def test_startup_runs_migration(self, mock_kb_manager_cls, mock_tenant_manager_cls):
        mock_kb_manager = mock_kb_manager_cls.return_value
        mock_tenant_manager = mock_tenant_manager_cls.return_value
        with TestClient(app):
            pass  # 进入/退出 with 块会触发 startup/shutdown 事件
        mock_kb_manager.migrate_json_if_needed.assert_called_once()
        mock_tenant_manager.migrate_default_tenant_if_needed.assert_called_once()
        mock_kb_manager.migrate_tenant_id_if_needed.assert_called_once_with(mock_tenant_manager)
        mock_kb_manager.migrate_vectorstore_layout_if_needed.assert_called_once()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 FastAPI 启动事件会调用一次 KBManager.migrate_json_if_needed()。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from ragify.api.main import app


class TestStartupMigration(unittest.TestCase):
    @patch("ragify.api.main.KBManager")
    def test_startup_runs_migration(self, mock_kb_manager_cls):
        mock_manager = mock_kb_manager_cls.return_value
        with TestClient(app):
            pass  # 进入/退出 with 块会触发 startup/shutdown 事件
        mock_manager.migrate_json_if_needed.assert_called_once()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块测试
验证 ConfigLoader 和 RAGifySettings 的行为。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.config.loader import ConfigLoader, RAGifySettings, initialize_config, get_config


class TestConfigLoader(unittest.TestCase):
    """测试 ConfigLoader"""

    def test_load_valid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("\n".join([
                "base:",
                "  project_name: TestProject",
                "llm:",
                "  provider: openai",
            ]))
            path = f.name
        try:
            loader = ConfigLoader(config_path=path)
            self.assertEqual(loader.get("base.project_name"), "TestProject")
            self.assertEqual(loader.get("llm.provider"), "openai")
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ConfigLoader(config_path="/nonexistent/config.yaml")

    def test_get_nested_key_with_default(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("base:\n  name: RAGify\n")
            path = f.name
        try:
            loader = ConfigLoader(config_path=path)
            self.assertEqual(loader.get("nonexistent.key", "default"), "default")
            self.assertIsNone(loader.get("nonexistent.key"))
        finally:
            os.unlink(path)

    def test_update_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("base:\n  name: RAGify\n")
            path = f.name
        try:
            loader = ConfigLoader(config_path=path)
            loader.update("base.name", "Updated")
            self.assertEqual(loader.get("base.name"), "Updated")
        finally:
            os.unlink(path)

    def test_env_var_resolution(self):
        os.environ["TEST_RAGIFY_KEY"] = "secret123"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("\n".join([
                "llm:",
                "  api_key: $TEST_RAGIFY_KEY",
            ]))
            path = f.name
        try:
            loader = ConfigLoader(config_path=path)
            self.assertEqual(loader.get("llm.api_key"), "secret123")
        finally:
            os.unlink(path)
            del os.environ["TEST_RAGIFY_KEY"]


class TestRAGifySettings(unittest.TestCase):
    """测试 RAGifySettings"""

    def test_default_values(self):
        settings = RAGifySettings()
        self.assertEqual(settings.project_name, "RAGify")
        self.assertEqual(settings.version, "0.1.0")
        self.assertEqual(settings.log_level, "INFO")

    def test_paths_become_absolute(self):
        settings = RAGifySettings(data_dir="./data", output_dir="./output")
        self.assertTrue(os.path.isabs(settings.data_dir))
        self.assertTrue(os.path.isabs(settings.output_dir))


class TestGlobalConfig(unittest.TestCase):
    """测试全局配置函数"""

    def test_get_config_returns_loader(self):
        initialize_config()
        config = get_config()
        self.assertIsInstance(config, ConfigLoader)


if __name__ == "__main__":
    unittest.main()

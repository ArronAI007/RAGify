#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块测试
"""

import os
import tempfile
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.config import ConfigLoader, RAGifySettings, get_config


class TestConfig(unittest.TestCase):
    """
    测试配置加载功能
    """
    
    def test_default_config_loading(self):
        """
        测试默认配置加载
        """
        try:
            config = get_config()
            self.assertIsInstance(config, RAGifySettings)
            self.assertEqual(config.basic.project_name, "RAGify")
            print("✓ 默认配置加载成功")
        except Exception as e:
            self.fail(f"默认配置加载失败: {str(e)}")
    
    def test_custom_config_loading(self):
        """
        测试自定义配置加载
        """
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
basic:
  project_name: "TestProject"
  log_level: "DEBUG"
  debug: true

llm:
  provider: "test_provider"
  model: "test_model"
""")
            temp_config_path = f.name
        
        try:
            loader = ConfigLoader(config_path=temp_config_path)
            config = loader.load_config()
            self.assertEqual(config.basic.project_name, "TestProject")
            self.assertEqual(config.basic.log_level, "DEBUG")
            self.assertEqual(config.llm.provider, "test_provider")
            print("✓ 自定义配置加载成功")
        except Exception as e:
            self.fail(f"自定义配置加载失败: {str(e)}")
        finally:
            # 清理临时文件
            os.unlink(temp_config_path)
    
    def test_config_validation(self):
        """
        测试配置验证功能
        """
        config = get_config()
        
        # 检查必要字段是否存在
        self.assertTrue(hasattr(config, 'basic'))
        self.assertTrue(hasattr(config, 'llm'))
        self.assertTrue(hasattr(config, 'embeddings'))
        self.assertTrue(hasattr(config, 'vectorstore'))
        
        # 检查类型是否正确
        self.assertIsInstance(config.basic.debug, bool)
        self.assertIsInstance(config.llm.temperature, (int, float))
        self.assertIsInstance(config.vectorstore.collection_name, str)
        
        print("✓ 配置验证成功")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent测试
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.agents import RAGifyAgent, RAGifyTool, RAGAgent, get_default_tools


class TestTool(RAGifyTool):
    """
    测试用的工具
    """
    name = "test_tool"
    description = "用于测试的工具"
    
    def _run(self, **kwargs):
        """
        运行工具
        """
        return f"Test tool executed with: {kwargs}"


class TestRAGifyAgent(unittest.TestCase):
    """
    测试Agent基类功能
    """
    
    def test_agent_registration(self):
        """
        测试Agent注册功能
        """
        from ragify.agents import AgentRegistry
        
        # 验证内置Agent已注册
        registry = AgentRegistry.get_instance()
        self.assertIn("RAGAgent", registry.agents)
        print("✓ Agent注册功能验证成功")
    
    def test_tool_creation(self):
        """
        测试工具创建
        """
        tool = TestTool()
        self.assertEqual(tool.name, "test_tool")
        result = tool.run(param1="value1")
        self.assertIn("param1", result)
        self.assertIn("value1", result)
        print("✓ 工具创建和运行成功")
    
    def test_default_tools(self):
        """
        测试默认工具集
        """
        tools = get_default_tools()
        self.assertTrue(len(tools) > 0)
        
        # 验证关键工具存在
        tool_names = [tool.name for tool in tools]
        self.assertIn("file_management", tool_names)
        print("✓ 默认工具集验证成功")


class TestRAGAgent(unittest.TestCase):
    """
    测试RAGAgent功能
    """
    
    @patch('ragify.agents.rag_agent.get_config')
    def test_rag_agent_initialization(self, mock_get_config):
        """
        测试RAGAgent初始化
        """
        # 模拟配置
        mock_config = MagicMock()
        mock_config.llm.provider = "mock_provider"
        mock_config.llm.model = "mock_model"
        mock_get_config.return_value = mock_config
        
        # 使用模拟的依赖初始化Agent
        with patch('ragify.agents.rag_agent.ChatOpenAI'), \
             patch('ragify.agents.rag_agent.OpenAIEmbeddings'), \
             patch('ragify.agents.rag_agent.ChromaDB'):
            
            agent = RAGAgent(name="test_rag_agent")
            self.assertEqual(agent.name, "test_rag_agent")
            print("✓ RAGAgent初始化成功")


if __name__ == "__main__":
    unittest.main()

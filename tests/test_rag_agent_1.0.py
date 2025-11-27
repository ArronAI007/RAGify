#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证适配LangChain 1.0 API的RAGAgent功能

该脚本测试RAGAgent的核心功能，包括初始化、invoke方法执行和工具调用等。
"""

import os
import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragify.config import get_config
from ragify.agents.rag_agent import RAGAgent
from ragify.agents.tools import create_custom_tool
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage


class TestRAGAgentLangChain1(unittest.TestCase):
    """测试RAGAgent在LangChain 1.0下的功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟配置对象（模拟get_config()的返回值）
        self.mock_config = MagicMock()
        self.mock_config.llm = MagicMock()
        self.mock_config.llm.provider = "mock"
        self.mock_config.llm.model = "mock-model"
        self.mock_config.vectorstore = MagicMock()
        self.mock_config.vectorstore.type = "mock"
        
        # 创建一个简单的测试工具
        def mock_tool_function(query: str) -> str:
            """模拟工具函数"""
            return f"Mock工具执行结果: {query}"
        
        self.test_tool = create_custom_tool(
            name="mock_tool",
            func=mock_tool_function,
            description="一个模拟工具，返回输入内容的修饰版本"
        )
    
    @patch('ragify.agents.rag_agent.QueryPipeline')
    @patch('ragify.agents.rag_agent.LanguageModelManager')
    @patch('ragify.agents.rag_agent.VectorStoreManager')
    def test_agent_initialization(self, mock_vector_store_manager, mock_language_model_manager, mock_query_pipeline):
        """测试Agent初始化是否成功"""
        # 配置模拟对象
        mock_llm = MagicMock()
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = mock_llm
        mock_language_model_manager.return_value = mock_lmm_instance
        
        mock_vector_store = MagicMock()
        mock_vector_store_manager.return_value.get_or_create.return_value = mock_vector_store
        
        mock_pipeline = MagicMock()
        mock_query_pipeline.return_value = mock_pipeline
        
        # 初始化Agent
        agent = RAGAgent(name="test_rag_agent", config=self.mock_config)
        
        # 记录初始工具数量
        initial_tool_count = len(agent.tools)
        
        # 添加测试工具
        agent.add_tools([self.test_tool])
        
        # 验证初始化
        self.assertIsNotNone(agent)
        self.assertEqual(len(agent.tools), initial_tool_count + 1)
        mock_language_model_manager.assert_called_once()
        mock_lmm_instance.get_llm.assert_called_once()
        mock_query_pipeline.assert_called_once()
        
    @patch('ragify.agents.rag_agent.QueryPipeline')
    @patch('ragify.agents.rag_agent.LanguageModelManager')
    @patch('ragify.agents.rag_agent.VectorStoreManager')
    def test_agent_invoke_method(self, mock_vector_store_manager, mock_language_model_manager, mock_query_pipeline):
        """测试Agent的invoke方法"""
        # 配置模拟对象
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "这是一个测试响应"
        mock_lmm_instance = MagicMock()
        mock_lmm_instance.get_llm.return_value = mock_llm
        mock_language_model_manager.return_value = mock_lmm_instance
        
        mock_vector_store = MagicMock()
        mock_vector_store_manager.return_value.get_or_create.return_value = mock_vector_store
        mock_vector_store.similarity_search.return_value = []
        
        mock_pipeline = MagicMock()
        mock_query_pipeline.return_value = mock_pipeline
        
        # 初始化Agent
        agent = RAGAgent(name="test_rag_agent", config=self.mock_config)
        agent.add_tools([self.test_tool])
        
        # 执行invoke
        result = agent.invoke("测试查询")
        
        # 验证结果
        self.assertIn("response", result)
        self.assertIn("execution_summary", result)
        self.assertGreaterEqual(result["execution_summary"]["execution_time"], 0)
        
    @patch('ragify.agents.rag_agent.QueryPipeline')
    @patch('ragify.agents.rag_agent.LanguageModelManager')
    @patch('ragify.agents.rag_agent.VectorStoreManager')
    def test_agent_with_tools(self, mock_vector_store_manager, mock_language_model_manager, mock_query_pipeline):
        """测试Agent是否能正确使用工具"""
        # 配置模拟对象
        mock_lmm_instance = MagicMock()
        mock_language_model_manager.return_value = mock_lmm_instance
        mock_llm = MagicMock()
        mock_lmm_instance.get_llm.return_value = mock_llm
        
        mock_vsm_instance = MagicMock()
        mock_vector_store_manager.return_value = mock_vsm_instance
        mock_vector_store = MagicMock()
        mock_vsm_instance.get_or_create.return_value = mock_vector_store
        mock_vector_store.similarity_search.return_value = []
        
        mock_pipeline = MagicMock()
        mock_query_pipeline.return_value = mock_pipeline
        
        # 初始化Agent
        agent = RAGAgent(name="test_rag_agent", config=self.mock_config)
        
        # 记录初始工具数量
        initial_tool_count = len(agent.tools)
        
        # 添加测试工具
        agent.add_tools([self.test_tool])
        
        # 验证工具被正确添加
        self.assertEqual(len(agent.tools), initial_tool_count + 1)
        
        # 模拟_rag_query方法直接返回结果
        with patch.object(agent, '_rag_query', return_value={
            "result": "模拟的RAG查询结果",
            "context": [],
            "stats": {"processing_time": 0.1}
        }) as mock_rag_query:
            
            # 执行Agent调用
            result = agent.invoke("测试查询")
            
            # 验证_rag_query被调用
            mock_rag_query.assert_called_once_with("测试查询")
            
            # 验证结果包含预期字段
            self.assertIn("query", result)
            self.assertIn("response", result)
            self.assertIn("execution_summary", result)
            
            # 验证response字段中包含预期数据
            self.assertIn("result", result["response"])
            self.assertIn("context", result["response"])
            self.assertIn("stats", result["response"])
            
    def test_tool_format_compatibility(self):
        """测试工具格式是否与LangChain 1.0兼容"""
        # 验证create_custom_tool创建的是Tool.from_function的实例
        self.assertIsInstance(self.test_tool, Tool)
        self.assertTrue(hasattr(self.test_tool, 'name'))
        self.assertTrue(hasattr(self.test_tool, 'description'))
        self.assertTrue(hasattr(self.test_tool, 'func'))


if __name__ == "__main__":
    # 运行所有测试
    unittest.main()

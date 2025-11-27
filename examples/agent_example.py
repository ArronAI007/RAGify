#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent示例
演示如何使用RAGify的Agent功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.agents import (RAGAgent, MultiModalRAGAgent, PipelineAgent,
                          get_default_tools, AgentRegistry)
from ragify.config import get_config
from ragify.mcp import IndexingPipeline


def main():
    """
    主函数，演示Agent的使用
    """
    print("===== RAGify Agent示例 =====")
    
    # 获取配置
    config = get_config()
    print(f"使用配置文件: {config.config_path}")
    
    # 确保示例文档已索引
    prepare_sample_data()
    
    # 1. 演示RAGAgent
    print("\n1. RAGAgent示例...")
    rag_agent = RAGAgent()
    
    rag_queries = [
        "什么是RAG系统，它有哪些核心组件？",
        "简要介绍LangChain 1.0的新特性。"
    ]
    
    for query in rag_queries:
        print(f"\n用户问题: {query}")
        print("-" * 60)
        response = rag_agent.ask(query)
        print(f"Agent回答: {response}")
    
    # 2. 演示带工具的RAGAgent
    print("\n\n2. 带工具的RAGAgent示例...")
    tools = get_default_tools()
    rag_agent_with_tools = RAGAgent(tools=tools)
    
    tool_queries = [
        "列出examples目录下的文件",
        "计算123乘以456的结果",
        "请总结RAG的优势"
    ]
    
    for query in tool_queries:
        print(f"\n用户问题: {query}")
        print("-" * 60)
        response = rag_agent_with_tools.ask(query)
        print(f"Agent回答: {response}")
    
    # 3. 演示PipelineAgent
    print("\n\n3. PipelineAgent示例...")
    pipeline_agent = PipelineAgent()
    
    pipeline_queries = [
        "执行索引操作，使用examples/sample_data目录",
        "查询有关多模态RAG的信息"
    ]
    
    for query in pipeline_queries:
        print(f"\n用户问题: {query}")
        print("-" * 60)
        response = pipeline_agent.ask(query)
        print(f"Agent回答: {response}")
    
    # 4. 演示可用的Agent类型
    print("\n\n4. 可用的Agent类型...")
    registry = AgentRegistry.get_registry()
    print(f"已注册的Agent类型: {', '.join(registry.list_agents())}")
    
    print("\n===== Agent示例完成 =====")


def prepare_sample_data():
    """
    准备示例数据，确保文档已索引
    """
    print("准备示例数据...")
    
    # 创建示例文档目录（如果基础示例脚本已经创建了这些文档）
    sample_data_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    
    # 检查目录是否存在，如果不存在，则从basic_rag_example导入创建函数
    if not os.path.exists(sample_data_dir) or not os.listdir(sample_data_dir):
        try:
            # 尝试导入创建示例文档的函数
            from examples.basic_rag_example import create_sample_documents
            create_sample_documents()
        except ImportError:
            print("警告: 无法导入创建示例文档的函数，将使用简单索引")
    
    # 执行索引操作
    print("开始索引示例文档...")
    index_pipeline = IndexingPipeline()
    
    try:
        index_result = index_pipeline.run({
            "directory_path": sample_data_dir,
            "clear_vectorstore": False  # 不强制清空，避免重复索引
        })
        summary = index_result.get("indexing_summary", {})
        print(f"索引完成，共索引 {summary.get('total_documents_indexed', 0)} 个文档")
    except Exception as e:
        print(f"索引过程中发生错误: {str(e)}")
        print("请确保配置文件正确，并且嵌入模型可用")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础RAG示例
演示如何使用RAGify进行文档索引和查询
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.mcp import IndexingPipeline, QueryPipeline
from ragify.config import get_config


def main():
    """
    主函数，演示基础RAG流程
    """
    print("===== RAGify 基础示例 =====")
    
    # 获取配置
    config = get_config()
    print(f"使用配置文件: {config.config_path}")
    
    # 创建示例文档（如果不存在）
    create_sample_documents()
    
    # 1. 索引文档
    print("\n1. 开始索引文档...")
    index_pipeline = IndexingPipeline()
    sample_data_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    
    index_result = index_pipeline.run({
        "directory_path": sample_data_dir,
        "clear_vectorstore": True  # 每次运行都清空向量存储
    })
    
    # 显示索引结果
    summary = index_result.get("indexing_summary", {})
    print("\n索引完成!")
    print(f"- 加载的文档总数: {summary.get('total_documents_loaded', 0)}")
    print(f"- 处理的文档总数: {summary.get('total_documents_processed', 0)}")
    print(f"- 生成的文档块总数: {summary.get('total_chunks_generated', 0)}")
    print(f"- 索引的文档总数: {summary.get('total_documents_indexed', 0)}")
    print(f"- 向量存储类型: {summary.get('vectorstore_type', 'unknown')}")
    
    # 2. 执行查询
    print("\n2. 开始查询示例...")
    query_pipeline = QueryPipeline()
    
    # 示例查询
    queries = [
        "什么是RAG系统？",
        "LangChain的主要功能有哪些？",
        "多模态RAG与传统RAG有什么不同？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n查询 {i}: {query}")
        print("-" * 60)
        
        # 执行查询
        query_result = query_pipeline.run({"query": query})
        
        # 显示结果
        response = query_result.get("response", "未生成响应")
        print(f"响应: {response}")
        
        # 显示检索到的文档信息
        retrieved_docs = query_result.get("retrieved_documents", [])
        print(f"\n检索到 {len(retrieved_docs)} 个相关文档:")
        for j, doc in enumerate(retrieved_docs[:3], 1):  # 只显示前3个
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("retrieval_score", "未知")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  来源 {j}: {source} (相关度: {score})")
            print(f"  内容预览: {content_preview}")
    
    print("\n===== 示例完成 =====")


def create_sample_documents():
    """
    创建示例文档用于演示
    """
    sample_data_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    os.makedirs(sample_data_dir, exist_ok=True)
    
    # 创建示例文档1
    doc1_path = os.path.join(sample_data_dir, "rag_introduction.txt")
    if not os.path.exists(doc1_path):
        with open(doc1_path, "w", encoding="utf-8") as f:
            f.write("""
# RAG系统介绍

检索增强生成(Retrieval-Augmented Generation, RAG)是一种结合了信息检索和生成式AI的技术。
它通过在生成回答之前检索相关文档来增强大型语言模型(LLM)的能力。

## RAG的核心组件

1. **文档加载器** - 负责从各种来源加载文档
2. **文档处理器** - 负责清理、分块和处理文档
3. **嵌入模型** - 将文本转换为向量表示
4. **向量数据库** - 存储文档向量
5. **检索器** - 检索与查询相关的文档
6. **生成模型** - 基于检索到的文档生成回答

## RAG的优势

- **知识更新** - 可以轻松更新知识库，无需重新训练模型
- **减少幻觉** - 通过引用可靠来源减少模型生成错误信息
- **可解释性** - 可以追踪回答的来源
- **领域适应性** - 可以针对特定领域进行定制

## RAG的应用场景

- **知识库问答**
- **客服系统**
- **技术文档助手**
- **个性化推荐**
- **研究辅助工具**
            """.strip())
    
    # 创建示例文档2
    doc2_path = os.path.join(sample_data_dir, "langchain_features.txt")
    if not os.path.exists(doc2_path):
        with open(doc2_path, "w", encoding="utf-8") as f:
            f.write("""
# LangChain 1.0 主要功能

LangChain是一个用于构建LLM应用的框架，它提供了一系列工具、组件和接口，帮助开发者快速构建复杂的AI应用。

## 核心组件

### 1. Chains
Chains允许将多个步骤组合在一起。它们可以是简单的链式调用，也可以是复杂的有条件逻辑的链。

### 2. Agents
Agents使用LLM来决定采取什么行动。有几种类型的Agents：
- **工具调用Agent** - 使用可用工具来回答问题
- **对话Agent** - 专注于与用户进行对话
- **规划Agent** - 能够规划和执行复杂任务

### 3. Memory
Memory组件用于在对话过程中保留状态。它可以存储：
- 整个对话历史
- 对话摘要
- 关键信息提取

### 4. 文档加载器
LangChain支持从多种来源加载文档：
- 文件系统(CSV, PDF, DOCX等)
- 网页
- 数据库
- API
- 云存储

### 5. 向量存储集成
LangChain与多种向量数据库集成：
- Chroma
- FAISS
- Pinecone
- Weaviate
- Qdrant

### 6. 回调系统
回调系统允许监控和记录LLM应用的执行过程。

## LangChain 1.0 的新特性

- **模块化设计** - 更灵活的组件架构
- **更好的TypeScript支持** - 改进的类型定义
- **新的Agent接口** - 更强大和灵活的Agent API
- **改进的文档加载器** - 支持更多格式和更好的性能
- **状态管理** - 更好的会话状态管理
- **错误处理** - 改进的错误处理和重试机制
            """.strip())
    
    # 创建示例文档3
    doc3_path = os.path.join(sample_data_dir, "multimodal_rag.txt")
    if not os.path.exists(doc3_path):
        with open(doc3_path, "w", encoding="utf-8") as f:
            f.write("""
# 多模态RAG

多模态RAG是传统RAG的扩展，它不仅处理文本，还可以处理图像、音频、视频等多种模态的内容。

## 多模态RAG与传统RAG的区别

### 输入多样性
- **传统RAG**: 仅接受文本查询
- **多模态RAG**: 接受文本、图像、音频等多种输入

### 文档类型
- **传统RAG**: 主要处理纯文本文档
- **多模态RAG**: 处理包含图像、表格、图表等混合内容的文档

### 嵌入技术
- **传统RAG**: 使用文本嵌入模型
- **多模态RAG**: 使用多模态嵌入模型，可以同时处理不同类型的数据

### 检索策略
- **传统RAG**: 基于文本相似度检索
- **多模态RAG**: 可以基于多模态相似度检索，例如文本查询匹配图像内容

## 多模态RAG的关键技术

### 1. 多模态嵌入
将不同类型的内容(文本、图像等)映射到同一个向量空间。

### 2. 跨模态检索
允许用一种模态的查询检索另一种模态的内容。

### 3. 多模态文档处理
能够处理和理解混合内容的文档。

### 4. 多模态生成
能够生成包含多种模态内容的响应。

## 应用场景

- **图像搜索与描述**
- **科学论文分析**（包含图表、公式）
- **产品目录查询**
- **医疗图像分析**
- **多媒体内容推荐**

## 技术挑战

- 不同模态之间的对齐
- 计算资源消耗大
- 评估标准复杂
- 多模态嵌入质量
            """.strip())
    
    print(f"示例文档已创建在: {sample_data_dir}")


if __name__ == "__main__":
    main()

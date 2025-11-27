#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGify CLI 命令行接口
提供项目的命令行操作功能
"""

import os
import sys
import click
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragify.config import get_config, ConfigLoader
from ragify.mcp import IndexingPipeline, QueryPipeline
# 导入更新后的agents模块
from ragify.core import VectorStoreManager


@click.group()
def ragify():
    """
    RAGify - 多模态RAG框架命令行工具
    """
    pass


@ragify.command()
def version():
    """
    显示RAGify版本信息
    """
    click.echo("RAGify v0.1.0")
    click.echo("一个强大、灵活的多模态RAG框架")


@ragify.command()
@click.argument("directory_path")
@click.option("--config", "-c", default=None, help="配置文件路径")
@click.option("--clear", is_flag=True, help="索引前清空向量存储")
def index(directory_path, config, clear):
    """
    索引指定目录下的文档
    """
    try:
        # 加载配置
        if config:
            config_loader = ConfigLoader(config_path=config)
            rag_config = config_loader.load_config()
        else:
            rag_config = get_config()
        
        click.echo(f"使用配置: {rag_config.config_path}")
        click.echo(f"开始索引目录: {directory_path}")
        
        # 执行索引
        pipeline = IndexingPipeline(config=rag_config)
        result = pipeline.run({
            "directory_path": directory_path,
            "clear_vectorstore": clear
        })
        
        # 显示结果
        summary = result.get("indexing_summary", {})
        click.echo("\n索引完成!")
        click.echo(f"- 加载的文档: {summary.get('total_documents_loaded', 0)}")
        click.echo(f"- 处理的文档: {summary.get('total_documents_processed', 0)}")
        click.echo(f"- 生成的块数: {summary.get('total_chunks_generated', 0)}")
        click.echo(f"- 索引的文档: {summary.get('total_documents_indexed', 0)}")
        
    except Exception as e:
        click.echo(f"索引过程中发生错误: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
@click.argument("query")
@click.option("--config", "-c", default=None, help="配置文件路径")
@click.option("--top-k", default=4, help="检索的文档数量")
def query(query, config, top_k):
    """
    执行查询并返回结果
    """
    try:
        # 加载配置
        if config:
            config_loader = ConfigLoader(config_path=config)
            rag_config = config_loader.load_config()
        else:
            rag_config = get_config()
        
        click.echo(f"使用配置: {rag_config.config_path}")
        click.echo(f"查询: {query}")
        
        # 执行查询
        pipeline = QueryPipeline(config=rag_config)
        result = pipeline.run({
            "query": query,
            "top_k": top_k
        })
        
        # 显示结果
        click.echo("\n查询结果:")
        click.echo("=" * 60)
        click.echo(result.get("response", "未生成响应"))
        click.echo("=" * 60)
        
        # 显示检索到的文档信息
        retrieved_docs = result.get("retrieved_documents", [])
        click.echo(f"\n检索到 {len(retrieved_docs)} 个相关文档:")
        for i, doc in enumerate(retrieved_docs[:3], 1):  # 只显示前3个
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("retrieval_score", "未知")
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            click.echo(f"\n来源 {i}: {source} (相关度: {score})")
            click.echo(f"内容预览: {content_preview}")
            
    except Exception as e:
        click.echo(f"查询过程中发生错误: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
@click.option("--config", "-c", default=None, help="配置文件路径")
def clear_index(config):
    """
    清空向量存储中的所有索引
    """
    try:
        # 加载配置
        if config:
            config_loader = ConfigLoader(config_path=config)
            rag_config = config_loader.load_config()
        else:
            rag_config = get_config()
        
        click.confirm("确定要清空向量存储中的所有数据吗？此操作不可撤销。", abort=True)
        
        # 清空向量存储
        vectorstore_manager = VectorStoreManager(config=rag_config)
        vectorstore_manager.clear()
        
        click.echo("向量存储已成功清空")
        
    except Exception as e:
        click.echo(f"清空向量存储时发生错误: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
@click.option("--config", "-c", default=None, help="配置文件路径")
def stats(config):
    """
    显示向量存储的统计信息
    """
    try:
        # 加载配置
        if config:
            config_loader = ConfigLoader(config_path=config)
            rag_config = config_loader.load_config()
        else:
            rag_config = get_config()
        
        # 获取统计信息
        vectorstore_manager = VectorStoreManager(config=rag_config)
        doc_count = vectorstore_manager.get_document_count()
        
        click.echo(f"向量存储统计信息:")
        click.echo(f"- 存储类型: {rag_config.vectorstore.type}")
        click.echo(f"- 集合名称: {rag_config.vectorstore.collection_name}")
        click.echo(f"- 持久化目录: {rag_config.vectorstore.persist_directory}")
        click.echo(f"- 文档数量: {doc_count}")
        
    except Exception as e:
        click.echo(f"获取统计信息时发生错误: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
def shell():
    """
    启动交互式shell
    """
    try:
        import code
        import readline
        import rlcompleter
        
        # 导入常用模块
        from ragify import config, core, mcp, agents
        
        # 设置交互式环境
        readline.parse_and_bind("tab: complete")
        
        # 定义欢迎信息和局部变量
        welcome = """
        RAGify 交互式Shell
        ------------------
        可用模块: config, core, mcp, agents
        可用函数: get_config, IndexingPipeline, QueryPipeline, RAGAgent
        示例: pipeline = mcp.IndexingPipeline(); pipeline.run({"directory_path": "./docs"})
        """
        
        # 安全导入Agent相关功能
        agent_dict = {}
        try:
            from ragify.agents import RAGAgent
            agent_dict["RAGAgent"] = RAGAgent
        except ImportError:
            click.echo("警告: 无法导入RAGAgent模块")
            
        local_vars = {
            "config": config,
            "core": core,
            "mcp": mcp,
            "agents": agents,
            "get_config": config.get_config,
            "IndexingPipeline": mcp.IndexingPipeline,
            "QueryPipeline": mcp.QueryPipeline,
            **agent_dict,
        }
        
        click.echo(welcome)
        code.interact(local=local_vars)
        
    except ImportError as e:
        click.echo(f"启动shell失败: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
@click.argument("command")
@click.option("--config", "-c", default=None, help="配置文件路径")
def agent(command, config):
    """
    使用RAGAgent执行命令
    """
    try:
        # 加载配置
        if config:
            config_loader = ConfigLoader(config_path=config)
            rag_config = config_loader.load_config()
        else:
            rag_config = get_config()
        
        click.echo(f"使用配置: {rag_config.config_path}")
        click.echo(f"Agent命令: {command}")
        
        # 创建Agent并执行命令（适配LangChain 1.0）
        try:
            from ragify.agents import RAGAgent, get_default_tools
            agent_instance = RAGAgent(config=rag_config, tools=get_default_tools())
            
            # 使用invoke方法替代ask方法
            result = agent_instance.invoke(command)
            response = result.get("response", "未生成响应")
            
            # 显示结果
            click.echo("\nAgent响应:")
            click.echo("=" * 60)
            click.echo(response)
            click.echo("=" * 60)
            
            # 显示执行摘要（如果有）
            if "execution_summary" in result:
                summary = result["execution_summary"]
                click.echo(f"\n执行统计:")
                click.echo(f"- 执行时间: {summary.get('execution_time', 0):.2f}秒")
                click.echo(f"- 工具调用次数: {summary.get('tool_calls_count', 0)}")
                
        except ImportError as e:
            click.echo(f"错误: 无法导入Agent模块: {str(e)}", err=True)
            click.echo("提示: 请检查agents模块是否正确安装和更新")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Agent执行时发生错误: {str(e)}", err=True)
        sys.exit(1)


@ragify.command()
@click.option("--output", "-o", default="config.yaml.example", help="输出文件路径")
def init_config(output):
    """
    生成示例配置文件
    """
    try:
        # 示例配置内容
        example_config = """
basic:
  project_name: "RAGify"
  log_level: "INFO"
  debug: false

llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 2048
  api_key_env: "OPENAI_API_KEY"

embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimensions: 1536
  api_key_env: "OPENAI_API_KEY"

vectorstore:
  type: "chromadb"
  persist_directory: "./vectorstore"
  collection_name: "default"

multimodal:
  enabled: true
  ocr_enabled: true
  image_processing:
    enabled: true
    max_size: 1024

mcp:
  enabled: true
  default_pipeline: "basic_rag"

agent:
  enabled: true
  default_agent: "rag_agent"

retrieval:
  k: 4
  fetch_k: 20
  score_threshold: 0.7

chains:
  type: "retrieval_qa"
  chain_type: "stuff"
"""
        
        # 写入文件
        with open(output, "w", encoding="utf-8") as f:
            f.write(example_config.strip())
        
        click.echo(f"示例配置文件已生成: {output}")
        click.echo("请根据需要修改配置文件，特别是API密钥设置")
        
    except Exception as e:
        click.echo(f"生成配置文件时发生错误: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    ragify()

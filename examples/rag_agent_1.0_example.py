#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例脚本：演示适配LangChain 1.0 API的RAGAgent使用方法

该脚本展示了如何初始化RAGAgent、使用内置工具以及执行查询。
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragify.config import get_config
from ragify.agents import RAGAgent, get_default_tools
from ragify.agents.tools import create_custom_tool


def custom_calculator_tool(expression: str) -> str:
    """自定义计算器工具示例"""
    try:
        # 注意：这里仅作为示例，实际使用时应使用更安全的计算方法
        result = eval(expression, {'__builtins__': {}})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def main():
    """主函数：演示RAGAgent的使用"""
    print("=" * 60)
    print("RAGAgent 1.0 API 使用示例")
    print("=" * 60)
    
    # 加载配置
    print("加载配置...")
    try:
        config = get_config()
        print(f"配置加载成功: {config.config_path}")
    except Exception as e:
        print(f"配置加载失败: {str(e)}")
        print("请确保配置文件正确配置了LLM和向量数据库设置")
        return
    
    # 获取默认工具并添加自定义工具
    print("\n初始化工具...")
    try:
        tools = get_default_tools()
        
        # 添加自定义计算器工具
        calculator_tool = create_custom_tool(
            name="calculator",
            func=custom_calculator_tool,
            description="计算数学表达式并返回结果"
        )
        tools.append(calculator_tool)
        
        print(f"工具初始化完成，共 {len(tools)} 个工具")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
    except Exception as e:
        print(f"工具初始化失败: {str(e)}")
        return
    
    # 初始化RAGAgent
    print("\n初始化RAGAgent...")
    try:
        start_time = time.time()
        agent = RAGAgent(config=config, tools=tools)
        init_time = time.time() - start_time
        print(f"RAGAgent初始化完成，耗时 {init_time:.2f} 秒")
    except Exception as e:
        print(f"RAGAgent初始化失败: {str(e)}")
        print("请检查LLM和向量数据库配置")
        return
    
    # 执行测试查询
    print("\n执行测试查询...")
    print("(输入 'exit' 退出程序)")
    print("=" * 60)
    
    # 预设的测试查询示例
    test_queries = [
        "简单介绍一下这个RAG系统",
        "使用calculator工具计算 10 + 20 * 3",
        "列出可用的工具及其功能",
    ]
    
    # 执行预设查询
    for i, query in enumerate(test_queries, 1):
        print(f"\n示例查询 {i}: {query}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            result = agent.invoke(query)
            end_time = time.time()
            
            # 显示结果
            print(f"响应: {result.get('response', '无响应')}")
            print(f"执行时间: {end_time - start_time:.2f} 秒")
            
            # 显示执行摘要
            if "execution_summary" in result:
                summary = result["execution_summary"]
                print(f"工具调用次数: {summary.get('tool_calls_count', 0)}")
                
        except Exception as e:
            print(f"查询执行失败: {str(e)}")
    
    print("\n交互式查询模式:")
    print("=" * 60)
    
    # 交互式查询
    while True:
        try:
            query = input("\n请输入查询 (exit to quit): ")
            if query.lower() == 'exit':
                break
                
            if not query.strip():
                continue
                
            start_time = time.time()
            result = agent.invoke(query)
            end_time = time.time()
            
            # 显示结果
            print(f"\n响应: {result.get('response', '无响应')}")
            print(f"执行时间: {end_time - start_time:.2f} 秒")
            
            # 显示执行摘要
            if "execution_summary" in result:
                summary = result["execution_summary"]
                print(f"工具调用次数: {summary.get('tool_calls_count', 0)}")
                
        except KeyboardInterrupt:
            print("\n程序中断")
            break
        except Exception as e:
            print(f"查询执行失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("程序结束")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本，用于验证RAGify的基本功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 只导入基本的配置模块来测试基本功能
from ragify.config import get_config


def main():
    """
    主函数，测试基本功能
    """
    print("===== RAGify 基础测试 =====")
    
    # 获取配置 - 这是最基本的功能
    try:
        config = get_config()
        print(f"✅ 成功加载配置文件: {config.config_path}")
        # 直接打印配置对象以了解其结构
        print(f"  配置对象类型: {type(config)}")
        # 尝试安全地访问配置属性
        if hasattr(config, 'get'):
            vectorstore_config = config.get('vectorstore', {})
            embeddings_config = config.get('embeddings', {})
            llm_config = config.get('llm', {})
            print(f"  向量存储类型: {vectorstore_config.get('type', '未知')}")
            print(f"  嵌入模型: {embeddings_config.get('model', '未知')}")
            print(f"  LLM模型: {llm_config.get('model', '未知')}")
        else:
            print("  注意: 配置对象不支持get方法")
    except Exception as e:
        print(f"❌ 加载配置失败: {str(e)}")
        return
    
    # 检查核心模块是否可以导入
    try:
        # 尝试导入核心组件但不初始化它们
        from ragify.core.vectorstores import VectorStoreManager
        print("✅ 成功导入VectorStoreManager")
    except Exception as e:
        print(f"❌ 导入VectorStoreManager失败: {str(e)}")
    
    try:
        from ragify.core.chains import RAGChainManager
        print("✅ 成功导入RAGChainManager")
    except Exception as e:
        print(f"❌ 导入RAGChainManager失败: {str(e)}")
    
    print("\n===== 测试完成 =====")
    print("注意: 此测试仅验证基本导入功能，不执行完整的RAG流程。")
    print("要修复完整的RAG流程，需要更新所有langchain导入以适应1.0版本。")


if __name__ == "__main__":
    main()

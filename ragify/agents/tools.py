from typing import Dict, Any, List, Optional, Callable
from langchain_core.tools import Tool
import os
import json
import logging
import requests
from datetime import datetime

from ..config import get_config
from ..mcp import IndexingPipeline


class FileManagementTool:
    """
    文件管理工具
    提供文件读取、写入、列表等功能
    """
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        if not os.path.exists(file_path):
            return f"错误: 文件 {file_path} 不存在"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取文件失败: {str(e)}"
    
    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """
        写入文件内容
        
        Args:
            file_path: 文件路径
            content: 文件内容
            
        Returns:
            操作结果
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"文件 {file_path} 写入成功"
        except Exception as e:
            return f"写入文件失败: {str(e)}"
    
    @staticmethod
    def list_files(directory: str, pattern: Optional[str] = None) -> str:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式（可选）
            
        Returns:
            文件列表
        """
        if not os.path.isdir(directory):
            return f"错误: 目录 {directory} 不存在或不是一个有效的目录"
        
        try:
            import glob
            
            if pattern:
                search_path = os.path.join(directory, pattern)
                files = glob.glob(search_path)
            else:
                files = os.listdir(directory)
                files = [os.path.join(directory, f) for f in files]
            
            # 过滤出文件（不包括目录）
            files = [f for f in files if os.path.isfile(f)]
            
            if not files:
                return f"目录 {directory} 中没有找到文件"
            
            result = f"在目录 {directory} 中找到 {len(files)} 个文件:\n"
            for file in files:
                file_size = os.path.getsize(file)
                modified_time = os.path.getmtime(file)
                modified_str = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
                result += f"- {file} (大小: {file_size} 字节, 修改时间: {modified_str})\n"
            
            return result
        except Exception as e:
            return f"列出文件失败: {str(e)}"
    
    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        获取文件管理工具列表
        
        Returns:
            Tool对象列表
        """
        return [
            Tool.from_function(
                name="read_file",
                func=cls.read_file,
                description="读取文件内容。输入应该是一个文件路径字符串。返回文件的完整内容。"
            ),
            Tool.from_function(
                name="write_file",
                func=cls.write_file,
                description="将内容写入文件。输入应该是一个包含文件路径和内容的字符串，格式为'file_path|||content'。返回操作结果。"
            ),
            Tool.from_function(
                name="list_files",
                func=cls.list_files,
                description="列出目录中的文件。输入应该是一个目录路径字符串。可选地，可以添加第二个参数作为文件匹配模式。返回找到的文件列表。"
            )
        ]


class IndexingTool:
    """
    索引管理工具
    提供文档索引相关功能
    """
    
    @staticmethod
    def index_directory(directory_path: str, clear_existing: bool = False) -> str:
        """
        索引指定目录中的文档
        
        Args:
            directory_path: 目录路径
            clear_existing: 是否清空现有索引
            
        Returns:
            索引结果
        """
        if not os.path.isdir(directory_path):
            return f"错误: 目录 {directory_path} 不存在或不是一个有效的目录"
        
        try:
            # 创建索引流水线
            pipeline = IndexingPipeline()
            
            # 运行索引流水线
            result = pipeline.run({
                "directory_path": directory_path,
                "clear_vectorstore": clear_existing
            })
            
            # 构建结果信息
            summary = result.get("indexing_summary", {})
            info = f"索引完成!\n"
            info += f"- 加载的文档总数: {summary.get('total_documents_loaded', 0)}\n"
            info += f"- 处理的文档总数: {summary.get('total_documents_processed', 0)}\n"
            info += f"- 生成的文档块总数: {summary.get('total_chunks_generated', 0)}\n"
            info += f"- 索引的文档总数: {summary.get('total_documents_indexed', 0)}\n"
            info += f"- 向量存储中文档总数: {summary.get('vectorstore_total_docs', 0)}\n"
            
            return info
        except Exception as e:
            return f"索引失败: {str(e)}"
    
    @staticmethod
    def index_file(file_path: str) -> str:
        """
        索引单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            索引结果
        """
        if not os.path.exists(file_path):
            return f"错误: 文件 {file_path} 不存在"
        
        try:
            # 创建索引流水线
            pipeline = IndexingPipeline()
            
            # 运行索引流水线
            result = pipeline.run({"file_paths": [file_path]})
            
            # 构建结果信息
            summary = result.get("indexing_summary", {})
            info = f"文件索引完成!\n"
            info += f"- 索引的文档总数: {summary.get('total_documents_indexed', 0)}\n"
            info += f"- 向量存储中文档总数: {summary.get('vectorstore_total_docs', 0)}\n"
            
            return info
        except Exception as e:
            return f"索引失败: {str(e)}"
    
    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        获取索引管理工具列表
        
        Returns:
            Tool对象列表
        """
        return [
            Tool.from_function(
                name="index_directory",
                func=cls.index_directory,
                description="索引指定目录中的所有文档。输入应该是一个目录路径字符串。可选地，可以添加第二个布尔参数表示是否清空现有索引。返回索引结果摘要。"
            ),
            Tool.from_function(
                name="index_file",
                func=cls.index_file,
                description="索引单个文件。输入应该是一个文件路径字符串。返回索引结果摘要。"
            )
        ]


class WebSearchTool:
    """
    网络搜索工具
    提供网络搜索功能
    注意：此工具需要配置API密钥
    """
    
    @staticmethod
    def search(query: str, max_results: int = 5) -> str:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果
        """
        try:
            # 获取配置
            config = get_config()
            search_config = config.get("web_search", {})
            
            # 检查是否配置了搜索引擎
            if not search_config.get("enabled", False):
                return "网络搜索功能未启用。请在配置文件中启用并配置搜索引擎。"
            
            # 在实际实现中，这里应该调用搜索引擎API
            # 例如Google Search API, Bing Search API等
            # 这里只是返回模拟结果
            
            return f"搜索结果 (模拟):\n"
            f"1. 结果标题 1 - 这是关于 '{query}' 的第一个搜索结果的摘要内容...\n"
            f"2. 结果标题 2 - 这是关于 '{query}' 的第二个搜索结果的摘要内容...\n"
            f"3. 结果标题 3 - 这是关于 '{query}' 的第三个搜索结果的摘要内容...\n"
            f"\n注意：这是模拟结果。要启用实际的网络搜索功能，请配置搜索引擎API密钥。"
            
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        获取网络搜索工具列表
        
        Returns:
            Tool对象列表
        """
        return [
            Tool.from_function(
                name="web_search",
                func=cls.search,
                description="执行网络搜索。输入应该是一个搜索查询字符串。返回搜索结果列表。"
            )
        ]


class CalculatorTool:
    """
    计算器工具
    提供数学计算功能
    """
    
    @staticmethod
    def calculate(expression: str) -> str:
        """
        计算数学表达式
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 安全计算数学表达式
            # 使用eval是有风险的，实际生产环境中应该使用更安全的计算方法
            # 这里为了演示目的使用eval，并限制只能使用数学运算
            
            # 定义允许的函数和变量
            import math
            allowed_names = {
                "abs": abs,
                "int": int,
                "float": float,
                "round": round,
                "math": math
            }
            
            # 移除可能的安全风险
            expression = expression.replace("import", "").replace("exec", "").replace("eval", "")
            
            # 计算表达式
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return f"计算结果: {expression} = {result}"
            
        except SyntaxError:
            return f"语法错误: 无法解析表达式 '{expression}'"
        except NameError as e:
            return f"名称错误: {str(e)}"
        except Exception as e:
            return f"计算失败: {str(e)}"
    
    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        获取计算器工具列表
        
        Returns:
            Tool对象列表
        """
        return [
            Tool.from_function(
                name="calculate",
                func=cls.calculate,
                description="计算数学表达式。输入应该是一个数学表达式字符串。返回计算结果。"
            )
        ]


class UtilityTool:
    """
    实用工具集
    提供各种实用功能
    """
    
    @staticmethod
    def format_json(json_str: str, indent: int = 2) -> str:
        """
        格式化JSON字符串
        
        Args:
            json_str: JSON字符串
            indent: 缩进空格数
            
        Returns:
            格式化后的JSON
        """
        try:
            # 解析JSON
            data = json.loads(json_str)
            # 格式化JSON
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {str(e)}"
        except Exception as e:
            return f"格式化失败: {str(e)}"
    
    @staticmethod
    def summarize_text(text: str, max_length: int = 100) -> str:
        """
        简单文本摘要
        
        Args:
            text: 要摘要的文本
            max_length: 摘要最大长度
            
        Returns:
            文本摘要
        """
        try:
            # 简单的文本摘要实现
            if len(text) <= max_length:
                return text
            
            # 查找合适的位置截断
            truncated = text[:max_length]
            last_space = truncated.rfind(' ')
            
            if last_space > 0:
                truncated = truncated[:last_space]
            
            return truncated + "..."
        except Exception as e:
            return f"摘要失败: {str(e)}"
    
    @classmethod
    def get_tools(cls) -> List[Tool]:
        """
        获取实用工具列表
        
        Returns:
            Tool对象列表
        """
        return [
            Tool.from_function(
                name="format_json",
                func=cls.format_json,
                description="格式化JSON字符串。输入应该是一个JSON字符串。返回格式化后的JSON。"
            ),
            Tool.from_function(
                name="summarize_text",
                func=cls.summarize_text,
                description="生成文本摘要。输入应该是要摘要的文本。返回截断的文本摘要。"
            )
        ]


def get_default_tools() -> List[Tool]:
    """
    获取所有默认工具
    
    Returns:
        默认工具列表
    """
    tools = []
    
    # 添加各种工具
    tools.extend(FileManagementTool.get_tools())
    tools.extend(IndexingTool.get_tools())
    tools.extend(CalculatorTool.get_tools())
    tools.extend(UtilityTool.get_tools())
    
    # 可选添加网络搜索工具
    config = get_config()
    if config.get("web_search", {}).get("enabled", False):
        tools.extend(WebSearchTool.get_tools())
    
    return tools


def create_custom_tool(name: str, func: Callable, description: str) -> Tool:
    """
    创建自定义工具
    
    Args:
        name: 工具名称
        func: 工具函数
        description: 工具描述
        
    Returns:
        Tool对象
    """
    return Tool.from_function(name=name, func=func, description=description)

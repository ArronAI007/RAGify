from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
import json
import logging

from .base import RAGifyAgent, register_agent, AgentExecutionObserver
from ..core import VectorStoreManager, LanguageModelManager
from ..mcp import QueryPipeline
from ..config import get_config


@register_agent("RAGAgent")
class RAGAgent(RAGifyAgent):
    """
    RAG Agent
    基于检索增强生成的Agent实现
    """
    
    def _initialize(self) -> None:
        """
        初始化RAG Agent
        """
        # 初始化核心组件
        self.vectorstore_manager = VectorStoreManager()
        self.language_model_manager = LanguageModelManager()
        
        # 获取语言模型
        self.llm = self.language_model_manager.get_llm()
        
        # 初始化查询流水线
        self.query_pipeline = QueryPipeline()
        
        # 添加默认工具
        self._setup_default_tools()
        
        # 初始化Agent执行器
        self._setup_agent_executor()
        
        # 初始化观察器
        self.observer = AgentExecutionObserver()
        
        self.logger.info(f"RAG Agent '{self.name}' 初始化完成")
    
    def _setup_default_tools(self) -> None:
        """
        设置默认工具
        """
        # 添加RAG查询工具
        self.add_tool({
            "name": "rag_query",
            "func": self._rag_query,
            "description": "用于回答与知识库相关的问题。输入应该是一个查询字符串。返回基于检索到的文档内容的答案。"
        })
        
        # 添加向量存储信息工具
        self.add_tool({
            "name": "vectorstore_info",
            "func": self._vectorstore_info,
            "description": "获取向量存储的基本信息，包括文档数量等统计数据。不需要参数。"
        })
    
    def _setup_agent_executor(self) -> None:
        """
        设置Agent执行器 (自定义实现，替代LangChain 1.0移除的AgentExecutor)
        """
        # 创建系统提示
        self.system_prompt = """你是一个基于检索增强生成(RAG)的智能助手。你的任务是帮助用户回答问题，解决问题，或者提供信息。

你可以使用以下工具：
1. rag_query - 用于回答与知识库相关的问题
2. vectorstore_info - 获取向量存储的基本信息

请根据用户的问题，决定使用哪个工具来获取信息。如果用户的问题与知识库内容相关，请使用rag_query工具。如果用户想了解知识库的状态，请使用vectorstore_info工具。

如果你无法使用工具回答，或者问题不需要工具就可以直接回答，请直接向用户回复。"""
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 初始化工具映射
        self.tool_map = {tool.name: tool for tool in self.tools}
    
    def _rag_query(self, query: str) -> str:
        """
        RAG查询工具函数
        
        Args:
            query: 用户查询
            
        Returns:
            查询结果
        """
        try:
            # 使用查询流水线处理
            result = self.query_pipeline.run({"query": query})
            
            if "response" in result:
                # 构建包含来源的完整响应
                response_text = result["response"]
                
                # 添加来源信息
                if "retrieved_documents" in result and result["retrieved_documents"]:
                    sources = []
                    for i, doc in enumerate(result["retrieved_documents"][:3], 1):
                        source = doc.metadata.get("source", "未知来源")
                        score = doc.metadata.get("retrieval_score", "未知")
                        sources.append(f"来源 {i}: {source} (相关度: {score})")
                    
                    if sources:
                        response_text += "\n\n参考来源:\n" + "\n".join(sources)
                
                return response_text
            else:
                return "抱歉，无法生成响应。请检查您的查询或系统配置。"
                
        except Exception as e:
            self.logger.error(f"RAG查询失败: {str(e)}")
            return f"查询处理过程中发生错误: {str(e)}"
    
    def _vectorstore_info(self) -> str:
        """
        向量存储信息工具函数
        
        Returns:
            向量存储信息
        """
        try:
            # 获取向量存储信息
            doc_count = self.vectorstore_manager.get_document_count()
            store_type = self.vectorstore_manager.vectorstore_type
            
            # 构建信息文本
            info = f"向量存储信息:\n"
            info += f"- 存储类型: {store_type}\n"
            info += f"- 文档总数: {doc_count}\n"
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取向量存储信息失败: {str(e)}")
            return f"获取向量存储信息过程中发生错误: {str(e)}"
    
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent调用 (自定义实现，替代LangChain 1.0移除的AgentExecutor)
        
        Args:
            query: 用户查询
            **kwargs: 额外参数
            
        Returns:
            调用结果字典
        """
        # 重置观察器
        self.observer.reset()
        
        # 开始执行观察
        self.observer.on_execution_start(query, **kwargs)
        
        try:
            # 自定义Agent执行逻辑
            # 1. 首先尝试直接回答
            # 2. 如果需要工具，通过LLM分析决定使用哪个工具
            
            # 直接使用RAG查询作为默认行为，简化实现
            # 这是一个简化版本，完整实现需要LLM分析决定是否调用工具
            result = self._rag_query(query)
            
            # 记录执行结束
            self.observer.on_execution_end({"output": result})
            
            # 构建完整结果
            response = {
                "query": query,
                "response": result,
                "execution_summary": self.observer.get_execution_summary()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Agent执行失败: {str(e)}")
            return {
                "query": query,
                "response": f"执行过程中发生错误: {str(e)}",
                "error": str(e)
            }


@register_agent("MultiModalRAGAgent")
class MultiModalRAGAgent(RAGAgent):
    """
    多模态RAG Agent
    支持处理包含图像等多模态内容的查询
    """
    
    def _initialize(self) -> None:
        """
        初始化多模态RAG Agent
        """
        super()._initialize()
        
        # 使用多模态查询流水线
        from ..mcp import MultiModalQueryPipeline
        self.query_pipeline = MultiModalQueryPipeline()
        
        # 添加多模态特定工具
        self.add_tool({
            "name": "multimodal_query",
            "func": self._multimodal_query,
            "description": "用于处理包含图像等多模态内容的查询。输入应该是一个包含'query'和可选的'image_urls'字段的JSON字符串。"
        })
        
        # 重新设置Agent执行器以包含新工具
        self._setup_agent_executor()
        
        self.logger.info(f"多模态RAG Agent '{self.name}' 初始化完成")
    
    def _setup_agent_executor(self) -> None:
        """
        设置多模态RAG Agent执行器 (自定义实现)
        """
        # 创建系统提示
        self.system_prompt = """你是一个支持多模态的检索增强生成(RAG)智能助手。你的任务是帮助用户回答问题，解决问题，或者提供信息。

你可以使用以下工具：
1. rag_query - 用于回答与知识库相关的纯文本问题
2. multimodal_query - 用于处理包含图像等多模态内容的查询
3. vectorstore_info - 获取向量存储的基本信息

请根据用户的问题，决定使用哪个工具来获取信息：
- 如果用户的问题仅包含文本且与知识库内容相关，请使用rag_query工具
- 如果用户的问题包含图像或其他多模态内容，请使用multimodal_query工具
- 如果用户想了解知识库的状态，请使用vectorstore_info工具

multimodal_query工具需要JSON格式的输入，包含'query'字段和可选的'image_urls'字段。

如果你无法使用工具回答，或者问题不需要工具就可以直接回答，请直接向用户回复。"""
        
        # 创建提示模板
        from langchain_core.prompts import ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 更新工具映射
        self.tool_map = {tool.name: tool for tool in self.tools}
    
    def _multimodal_query(self, input_json: str) -> str:
        """
        多模态查询工具函数
        
        Args:
            input_json: 包含查询和可选图像URL的JSON字符串
            
        Returns:
            查询结果
        """
        try:
            # 解析JSON输入
            input_data = json.loads(input_json)
            query = input_data.get("query", "")
            image_urls = input_data.get("image_urls", [])
            
            if not query:
                return "错误: 查询内容不能为空"
            
            # 构建查询数据
            query_data = {"query": query}
            if image_urls:
                query_data["image_urls"] = image_urls
            
            # 使用多模态查询流水线处理
            result = self.query_pipeline.run(query_data)
            
            if "response" in result:
                return result["response"]
            else:
                return "抱歉，无法生成多模态响应。请检查您的输入或系统配置。"
                
        except json.JSONDecodeError:
            return "错误: 无效的JSON格式。请提供正确的JSON格式输入。"
        except Exception as e:
            self.logger.error(f"多模态查询失败: {str(e)}")
            return f"多模态查询处理过程中发生错误: {str(e)}"
    
    def invoke(self, query: str, image_urls: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        执行多模态Agent调用
        
        Args:
            query: 用户查询
            image_urls: 图像URL列表
            **kwargs: 额外参数
            
        Returns:
            调用结果字典
        """
        if image_urls:
            # 对于包含图像的查询，构建JSON输入
            input_data = {
                "query": query,
                "image_urls": image_urls
            }
            query = json.dumps(input_data)
        
        # 调用父类的invoke方法
        return super().invoke(query, **kwargs)


@register_agent("PipelineAgent")
class PipelineAgent(RAGifyAgent):
    """
    流水线Agent
    直接使用MCP流水线处理查询，不使用LangChain Agent框架
    更加轻量级和直接
    """
    
    def _initialize(self) -> None:
        """
        初始化流水线Agent
        """
        # 初始化查询流水线
        from ..mcp import QueryPipeline
        self.query_pipeline = QueryPipeline()
        
        self.logger.info(f"流水线Agent '{self.name}' 初始化完成")
    
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行Agent调用
        
        Args:
            query: 用户查询
            **kwargs: 额外参数，将传递给查询流水线
            
        Returns:
            调用结果字典
        """
        try:
            # 构建查询数据
            query_data = {"query": query, **kwargs}
            
            # 执行查询流水线
            result = self.query_pipeline.run(query_data)
            
            # 构建响应
            response = {
                "query": query,
                "response": result.get("response", "未生成响应"),
                "retrieved_documents": result.get("retrieved_documents", []),
                "query_summary": result.get("query_summary", {})
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"流水线执行失败: {str(e)}")
            return {
                "query": query,
                "response": f"执行过程中发生错误: {str(e)}",
                "error": str(e)
            }

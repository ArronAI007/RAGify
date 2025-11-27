from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..config import get_config


class LanguageModelManager:
    """
    语言模型管理器，负责管理LLM和生成响应
    """
    
    def __init__(self):
        self.config = get_config()
        self.llm_provider = self.config.get("llm.provider", "openai")
        self.llm_model = self.config.get("llm.model_name", "gpt-4")
        self.temperature = self.config.get("llm.temperature", 0.7)
        self.max_tokens = self.config.get("llm.max_tokens", 1024)
        
        # 初始化LLM
        self.llm = self._initialize_llm()
        self.output_parser = StrOutputParser()
    
    def _initialize_llm(self) -> BaseLanguageModel:
        """
        初始化语言模型
        """
        try:
            if self.llm_provider == "openai":
                # 获取API密钥
                api_key = self.config.get("llm.api_key")
                return ChatOpenAI(
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key,
                    streaming=True
                )
            
            elif self.llm_provider == "anthropic":
                # Anthropic Claude
                try:
                    api_key = self.config.get("llm.api_key")
                    return ChatAnthropic(
                        model=self.llm_model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        api_key=api_key
                    )
                except ImportError:
                    raise ImportError("请安装langchain-anthropic: pip install langchain-anthropic")
            
            else:
                # 默认使用OpenAI
                print(f"警告: 不支持的LLM提供者 {self.llm_provider}，使用OpenAI默认值")
                return ChatOpenAI(
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=True
                )
        
        except Exception as e:
            print(f"初始化语言模型时出错: {e}")
            # 返回None，稍后在调用时处理
            return None
    
    def generate_response(self, prompt: Union[str, List[BaseMessage]], **kwargs) -> str:
        """
        生成文本响应
        """
        if not self.llm:
            print("警告: 语言模型未初始化")
            return "语言模型未初始化，无法生成响应"
        
        try:
            if isinstance(prompt, str):
                # 简单字符串提示
                response = self.llm.invoke(prompt, **kwargs)
                return self._extract_content(response)
            elif isinstance(prompt, list):
                # 消息列表
                response = self.llm.invoke(prompt, **kwargs)
                return self._extract_content(response)
            else:
                raise ValueError(f"不支持的提示类型: {type(prompt)}")
        
        except Exception as e:
            print(f"生成响应时出错: {e}")
            return f"生成响应时出错: {str(e)}"
    
    def _extract_content(self, response: Any) -> str:
        """
        从LLM响应中提取内容
        """
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def create_rag_prompt(self, query: str, context_docs: List[Any]) -> str:
        """
        创建RAG提示模板
        """
        context_text = "\n\n".join([
            f"内容: {doc.page_content}\n来源: {doc.metadata.get('source', 'unknown')}" 
            for doc in context_docs
        ])
        
        prompt = f"""
        你是一个基于检索增强生成(RAG)的AI助手。请根据提供的上下文信息回答用户的问题。
        
        上下文信息:
        {context_text}
        
        用户问题:
        {query}
        
        请确保你的回答:
        1. 严格基于提供的上下文信息
        2. 对问题提供准确、全面的回答
        3. 如果上下文信息不足以回答问题，请明确说明
        4. 保持回答的逻辑性和可读性
        """
        
        return prompt
    
    def create_chat_prompt(self, query: str, chat_history: List[Dict[str, str]], context_docs: List[Any] = None) -> List[BaseMessage]:
        """
        创建聊天提示，支持历史记录和上下文
        """
        messages = []
        
        # 添加系统消息
        system_prompt = "你是一个基于检索增强生成(RAG)的AI助手。请使用提供的上下文信息回答用户的问题。"
        messages.append(SystemMessage(content=system_prompt))
        
        # 添加上下文（如果有）
        if context_docs:
            context_text = "\n\n".join([
                f"内容: {doc.page_content}\n来源: {doc.metadata.get('source', 'unknown')}" 
                for doc in context_docs
            ])
            context_message = f"参考信息:\n{context_text}"
            messages.append(SystemMessage(content=context_message))
        
        # 添加聊天历史
        for item in chat_history:
            if "user" in item:
                messages.append(HumanMessage(content=item["user"]))
            if "assistant" in item:
                messages.append(AIMessage(content=item["assistant"]))
        
        # 添加用户当前问题
        messages.append(HumanMessage(content=query))
        
        return messages


class MultiModalLanguageModelManager(LanguageModelManager):
    """
    多模态语言模型管理器，支持文本和图像等多模态输入
    """
    
    def __init__(self):
        super().__init__()
        self.multimodal_enabled = self.config.get("multimodal.enabled", True)
    
    def create_multimodal_prompt(self, query: str, context_docs: List[Any], images: List[Dict[str, Any]] = None) -> List[BaseMessage]:
        """
        创建多模态提示，支持文本和图像
        """
        if not self.multimodal_enabled:
            # 如果多模态未启用，回退到文本提示
            return self.create_chat_prompt(query, [], context_docs)
        
        messages = []
        
        # 添加系统消息
        system_prompt = "你是一个多模态AI助手，可以理解文本和图像。请根据提供的信息回答用户的问题。"
        messages.append(SystemMessage(content=system_prompt))
        
        # 添加文本上下文
        if context_docs:
            context_text = "\n\n".join([
                f"内容: {doc.page_content}\n来源: {doc.metadata.get('source', 'unknown')}" 
                for doc in context_docs
            ])
            messages.append(SystemMessage(content=f"文本参考信息:\n{context_text}"))
        
        # 创建用户消息，包含文本和图像
        user_content = []
        
        # 添加文本查询
        user_content.append({"type": "text", "text": query})
        
        # 添加图像（如果有）
        if images:
            for image_info in images:
                if "image_url" in image_info:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_info["image_url"]}
                    })
                elif "base64_image" in image_info:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_info['base64_image']}"}
                    })
        
        # 创建多模态用户消息
        messages.append(HumanMessage(content=user_content))
        
        return messages
    
    def generate_multimodal_response(self, query: str, context_docs: List[Any] = None, images: List[Dict[str, Any]] = None, **kwargs) -> str:
        """
        生成多模态响应
        """
        if not self.multimodal_enabled or not self.llm:
            # 如果多模态未启用或LLM未初始化，回退到文本响应
            if context_docs:
                prompt = self.create_rag_prompt(query, context_docs)
            else:
                prompt = query
            return self.generate_response(prompt, **kwargs)
        
        # 创建多模态提示
        messages = self.create_multimodal_prompt(query, context_docs, images)
        
        try:
            # 生成响应
            response = self.llm.invoke(messages, **kwargs)
            return self._extract_content(response)
        
        except Exception as e:
            print(f"生成多模态响应时出错: {e}")
            # 回退到文本响应
            return self.generate_response(query, **kwargs)

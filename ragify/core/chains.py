from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..config import get_config
from .vectorstores import VectorStoreManager
from .language_models import LanguageModelManager


class RAGChainManager:
    """
    RAG链管理器，负责创建和管理各种RAG链
    """
    
    def __init__(self):
        self.config = get_config()
        self.vectorstore_manager = VectorStoreManager()
        self.language_model_manager = LanguageModelManager()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化成功标志
        self.initialized = True
        print("RAG链管理器初始化成功")
    
    def run_qa_chain(self, query: str) -> Dict[str, Any]:
        """
        运行检索QA链
        """
        # 直接使用简单的RAG查询实现
        return self._simple_rag_query(query)
    
    def run_conversational_chain(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        运行会话检索链
        """
        try:
            # 使用简单的RAG查询实现，传入聊天历史
            result = self._simple_rag_query(query, chat_history)
            
            # 更新内存
            if result.get("answer"):
                self.memory.save_context({"input": query}, {"response": result["answer"]})
            
            return result
        except Exception as e:
            print(f"运行会话检索链时出错: {e}")
            # 回退到简单检索
            return self._simple_rag_query(query)
    
    def _simple_rag_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        简单的RAG查询实现
        """
        try:
            # 1. 检索相关文档
            k = self.config.get("retrieval.k", 3)
            score_threshold = self.config.get("retrieval.score_threshold")
            
            if score_threshold:
                docs_with_scores = self.vectorstore_manager.similarity_search_with_score(
                    query=query, 
                    k=k, 
                    score_threshold=score_threshold
                )
                docs = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
            else:
                docs = self.vectorstore_manager.similarity_search(
                    query=query, 
                    k=k
                )
                scores = None
            
            # 2. 生成提示
            prompt = self.language_model_manager.create_rag_prompt(query, docs)
            
            # 3. 生成响应
            answer = self.language_model_manager.generate_response(prompt)
            
            # 4. 构建结果
            result = {
                "query": query,
                "answer": answer,
                "source_documents": docs
            }
            
            if scores:
                result["scores"] = scores
            
            return result
        
        except Exception as e:
            print(f"执行简单RAG查询时出错: {e}")
            return {
                "query": query,
                "answer": f"处理查询时出错: {str(e)}",
                "source_documents": []
            }
    
    def update_retriever(self) -> None:
        """
        更新检索器
        当向量存储内容发生变化时调用
        """
        try:
            # 在这个简化版本中，我们不需要更新检索器，因为每次查询都会直接调用向量存储管理器
            print("检索器已更新")
        
        except Exception as e:
            print(f"更新检索器时出错: {e}")


class MultiModalRAGChainManager(RAGChainManager):
    """
    多模态RAG链管理器，支持文本和图像等多模态内容
    """
    
    def __init__(self):
        # 使用多模态语言模型管理器
        from .language_models import MultiModalLanguageModelManager
        
        self.config = get_config()
        self.vectorstore_manager = VectorStoreManager()
        self.language_model_manager = MultiModalLanguageModelManager()
        self.multimodal_enabled = self.config.get("multimodal.enabled", True)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 初始化成功标志
        self.initialized = True
        print("多模态RAG链管理器初始化成功")
    
    def run_multimodal_query(self, query: str, images: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        运行多模态查询
        """
        if not self.multimodal_enabled:
            # 如果多模态未启用，回退到常规RAG查询
            return self.run_qa_chain(query)
        
        try:
            # 1. 检索相关文档
            k = self.config.get("retrieval.k", 3)
            docs = self.vectorstore_manager.similarity_search(
                query=query, 
                k=k
            )
            
            # 2. 生成多模态响应
            answer = self.language_model_manager.generate_multimodal_response(
                query=query,
                context_docs=docs,
                images=images
            )
            
            # 3. 构建结果
            result = {
                "query": query,
                "answer": answer,
                "source_documents": docs,
                "multimodal": True
            }
            
            return result
        
        except Exception as e:
            print(f"执行多模态查询时出错: {e}")
            # 回退到常规查询
            return self.run_qa_chain(query)
    
    def run_multimodal_conversational_query(self, query: str, images: List[Dict[str, Any]] = None, 
                                          chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        运行多模态会话查询
        """
        if not self.multimodal_enabled:
            # 如果多模态未启用，回退到常规会话查询
            return self.run_conversational_chain(query, chat_history)
        
        # 对于多模态会话，我们需要特殊处理
        # 这里简单地将当前的多模态查询作为一次交互处理
        result = self.run_multimodal_query(query, images)
        
        # 如果提供了聊天历史，可以在这里更新
        if result.get("answer"):
            self.memory.save_context({"input": query}, {"response": result["answer"]})
        
        return result

from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from ..config import get_config
from .embeddings import EmbeddingGenerator


class VectorStoreManager:
    """
    向量存储管理器，负责文档的向量化存储和检索
    """
    
    def __init__(self):
        self.config = get_config()
        self.vectorstore_type = self.config.get("vectorstore.type", "chromadb")
        self.persist_directory = self.config.get("vectorstore.persist_directory", "./vectorstore")
        self.collection_name = self.config.get("vectorstore.collection_name", "ragify_documents")
        
        # 初始化嵌入生成器
        self.embedding_generator = EmbeddingGenerator()
        
        # 初始化向量存储
        self.vectorstore = self._initialize_vectorstore()
    
    def _initialize_vectorstore(self) -> Optional[VectorStore]:
        """
        初始化向量存储
        """
        try:
            if self.vectorstore_type == "chromadb":
                # 尝试加载现有的Chroma向量存储
                try:
                    vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        collection_name=self.collection_name,
                        embedding_function=self.embedding_generator.embeddings
                    )
                    print(f"成功加载现有的Chroma向量存储，集合: {self.collection_name}")
                    return vectorstore
                except Exception:
                    # 如果加载失败，创建新的
                    print(f"创建新的Chroma向量存储，集合: {self.collection_name}")
                    return Chroma(
                        persist_directory=self.persist_directory,
                        collection_name=self.collection_name,
                        embedding_function=self.embedding_generator.embeddings
                    )
            
            elif self.vectorstore_type == "faiss":
                # FAISS向量存储
                try:
                    vectorstore = FAISS.load_local(
                        self.persist_directory,
                        self.embedding_generator.embeddings
                    )
                    print("成功加载现有的FAISS向量存储")
                    return vectorstore
                except Exception:
                    # 创建新的FAISS存储
                    print("创建新的FAISS向量存储")
                    # FAISS需要先有文档才能创建，这里返回None
                    return None
            
            else:
                print(f"不支持的向量存储类型: {self.vectorstore_type}，使用ChromaDB默认值")
                return Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_generator.embeddings
                )
        
        except Exception as e:
            print(f"初始化向量存储时出错: {e}")
            return None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        向向量存储添加文档
        """
        if not documents:
            return []
        
        try:
            # 提取文本内容
            texts = [doc.page_content for doc in documents]
            
            # 为每个文档生成嵌入
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # 过滤掉无法生成嵌入的文档
            valid_docs_and_embeddings = []
            for doc, embedding in zip(documents, embeddings):
                if embedding:
                    valid_docs_and_embeddings.append((doc, embedding))
            
            if not valid_docs_and_embeddings:
                print("警告: 没有有效的文档可添加到向量存储")
                return []
            
            valid_docs, valid_embeddings = zip(*valid_docs_and_embeddings)
            
            # 添加到向量存储
            if self.vectorstore_type == "faiss":
                if self.vectorstore is None:
                    # 首次创建FAISS向量存储
                    self.vectorstore = FAISS.from_documents(
                        list(valid_docs),
                        self.embedding_generator.embeddings
                    )
                else:
                    # 添加到现有FAISS
                    self.vectorstore.add_documents(list(valid_docs))
                
                # 保存FAISS索引
                self.vectorstore.save_local(self.persist_directory)
            
            else:
                # Chroma或其他
                doc_ids = self.vectorstore.add_documents(list(valid_docs))
                
                # 持久化Chroma
                if self.vectorstore_type == "chromadb":
                    self.vectorstore.persist()
                
                print(f"成功添加 {len(valid_docs)} 个文档到向量存储")
                return doc_ids
            
        except Exception as e:
            print(f"添加文档到向量存储时出错: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        """
        执行相似度搜索
        """
        if not self.vectorstore:
            print("警告: 向量存储未初始化")
            return []
        
        try:
            # 从配置获取默认k值
            if k is None:
                k = self.config.get("retrieval.k", 3)
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                **kwargs
            )
            
            return results
        
        except Exception as e:
            print(f"相似度搜索时出错: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3, score_threshold: float = None) -> List[tuple[Document, float]]:
        """
        执行带分数的相似度搜索
        """
        if not self.vectorstore:
            print("警告: 向量存储未初始化")
            return []
        
        try:
            # 从配置获取默认值
            if k is None:
                k = self.config.get("retrieval.k", 3)
            if score_threshold is None:
                score_threshold = self.config.get("retrieval.score_threshold")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # 应用分数阈值过滤
            if score_threshold is not None:
                # 注意：Chroma和FAISS的分数含义不同
                # Chroma使用cosine距离（越低越相似），FAISS使用cosine相似度（越高越相似）
                if self.vectorstore_type == "chromadb":
                    results = [(doc, score) for doc, score in results if score <= (1 - score_threshold)]
                else:
                    results = [(doc, score) for doc, score in results if score >= score_threshold]
            
            return results
        
        except Exception as e:
            print(f"带分数的相似度搜索时出错: {e}")
            return []
    
    def clear(self) -> bool:
        """
        清除向量存储中的所有文档
        """
        try:
            if self.vectorstore_type == "chromadb":
                # Chroma需要重新创建集合
                import shutil
                import os
                
                # 关闭当前连接
                self.vectorstore = None
                
                # 删除持久化目录
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                
                # 重新初始化
                self.vectorstore = self._initialize_vectorstore()
            
            elif self.vectorstore_type == "faiss":
                # FAISS需要重新创建
                self.vectorstore = None
                
                # 删除索引文件
                import os
                for file in ["index.faiss", "index.pkl"]:
                    file_path = os.path.join(self.persist_directory, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            print("向量存储已清空")
            return True
        
        except Exception as e:
            print(f"清空向量存储时出错: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        获取向量存储中的文档数量
        """
        try:
            # 这个功能在不同向量存储中实现方式不同
            # 这里使用简单的方法获取
            if self.vectorstore_type == "chromadb":
                # ChromaDB
                collection = self.vectorstore.get()
                return len(collection.get("ids", []))
            elif self.vectorstore_type == "faiss":
                # FAISS - 只能通过搜索获取近似数量
                # 这不是精确计数，但在大多数情况下足够
                if self.vectorstore:
                    return len(self.vectorstore.index_to_docstore_id)
                return 0
            else:
                return 0
        except Exception as e:
            print(f"获取文档数量时出错: {e}")
            return 0

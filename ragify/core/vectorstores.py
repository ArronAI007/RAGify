from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from ..config import get_config
from .embeddings import EmbeddingGenerator
import logging

logger = logging.getLogger("ragify.core.vectorstores")


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
                    logger.info(f"成功加载现有的Chroma向量存储，集合: {self.collection_name}")
                    return vectorstore
                except Exception:
                    # 如果加载失败，创建新的
                    logger.info(f"创建新的Chroma向量存储，集合: {self.collection_name}")
                    return Chroma(
                        persist_directory=self.persist_directory,
                        collection_name=self.collection_name,
                        embedding_function=self.embedding_generator.embeddings
                    )
            
            elif self.vectorstore_type == "faiss":
                # FAISS向量存储
                import os
                index_file = os.path.join(self.persist_directory, "index.faiss")
                pkl_file = os.path.join(self.persist_directory, "index.pkl")
                if os.path.exists(index_file) and os.path.exists(pkl_file):
                    try:
                        vectorstore = FAISS.load_local(
                            self.persist_directory,
                            self.embedding_generator.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        logger.info("成功加载现有的FAISS向量存储")
                        return vectorstore
                    except Exception as e:
                        logger.error(f"加载FAISS向量存储失败: {e}")
                        logger.info("将创建新的FAISS向量存储")
                        return None
                else:
                    logger.info("FAISS索引文件不存在，将创建新的向量存储")
                    return None
            
            else:
                logger.info(f"不支持的向量存储类型: {self.vectorstore_type}，使用ChromaDB默认值")
                return Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_generator.embeddings
                )
        
        except Exception as e:
            logger.error(f"初始化向量存储时出错: {e}")
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
                logger.warning("警告: 没有有效的文档可添加到向量存储")
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
                logger.info(f"成功添加 {len(valid_docs)} 个文档到FAISS向量存储")
                return list(range(len(valid_docs)))

            else:
                # Chroma或其他
                doc_ids = self.vectorstore.add_documents(list(valid_docs))
                
                # 持久化Chroma
                if self.vectorstore_type == "chromadb":
                    self.vectorstore.persist()
                
                logger.info(f"成功添加 {len(valid_docs)} 个文档到向量存储")
                return doc_ids
            
        except Exception as e:
            logger.error(f"添加文档到向量存储时出错: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        """
        执行相似度搜索
        """
        if not self.vectorstore:
            logger.warning("警告: 向量存储未初始化")
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
            logger.error(f"相似度搜索时出错: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3, score_threshold: float = None) -> List[tuple[Document, float]]:
        """
        执行带分数的相似度搜索
        """
        if not self.vectorstore:
            logger.warning("警告: 向量存储未初始化")
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
            logger.error(f"带分数的相似度搜索时出错: {e}")
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
            
            logger.info("向量存储已清空")
            return True
        
        except Exception as e:
            logger.error(f"清空向量存储时出错: {e}")
            return False
    
    def get_sources(self) -> list[dict[str, str]]:
        """获取已索引文档的来源信息"""
        try:
            if not self.vectorstore:
                return []
            if self.vectorstore_type == "faiss":
                sources: dict[str, dict[str, str]] = {}
                for doc_id in self.vectorstore.index_to_docstore_id.values():
                    doc = self.vectorstore.docstore.search(doc_id)
                    if doc is None:
                        continue
                    src = doc.metadata.get("source", "")
                    if src and src not in sources:
                        sources[src] = {
                            "name": src.split("/")[-1],
                            "source": src,
                            "file_type": doc.metadata.get("file_type", ""),
                            "chunks": 1,
                        }
                    elif src in sources:
                        sources[src]["chunks"] += 1
                return list(sources.values())
            elif self.vectorstore_type == "chromadb":
                collection = self.vectorstore.get()
                sources = {}
                metadatas = collection.get("metadatas", [])
                for meta in metadatas or []:
                    src = meta.get("source", "")
                    if src and src not in sources:
                        sources[src] = {
                            "name": src.split("/")[-1],
                            "source": src,
                            "file_type": meta.get("file_type", ""),
                            "chunks": 1,
                        }
                    elif src in sources:
                        sources[src]["chunks"] += 1
                return list(sources.values())
            return []
        except Exception as e:
            logger.error(f"获取文档来源时出错: {e}")
            return []

    def get_document_count(self) -> int:
        """
        获取向量存储中的唯一源文件数量（非分块数）
        """
        try:
            sources = self.get_sources()
            return len(sources)
        except Exception as e:
            logger.error(f"获取文档数量时出错: {e}")
            return 0

    def get_chunks_by_source(self, source: str) -> list[dict]:
        """获取指定源文件的所有分块内容及ID"""
        chunks: list[dict] = []
        try:
            if not self.vectorstore:
                return chunks
            if self.vectorstore_type == "faiss":
                for idx, doc_id in self.vectorstore.index_to_docstore_id.items():
                    doc = self.vectorstore.docstore.search(doc_id)
                    if doc is None:
                        continue
                    if doc.metadata.get("source") == source:
                        chunks.append({
                            "chunk_id": str(idx),
                            "content": doc.page_content,
                            "metadata": {
                                "source": doc.metadata.get("source", ""),
                                "file_type": doc.metadata.get("file_type", ""),
                            },
                        })
            elif self.vectorstore_type == "chromadb":
                collection = self.vectorstore.get()
                ids_list = collection.get("ids", [])
                docs_list = collection.get("documents", [])
                metas = collection.get("metadatas", [])
                for i, doc_id in enumerate(ids_list):
                    meta = metas[i] if metas and i < len(metas) else {}
                    if meta.get("source") == source:
                        chunks.append({
                            "chunk_id": str(doc_id),
                            "content": docs_list[i] if docs_list else "",
                            "metadata": {
                                "source": meta.get("source", ""),
                                "file_type": meta.get("file_type", ""),
                            },
                        })
            return chunks
        except Exception as e:
            logger.error(f"获取分块时出错: {e}")
            return chunks

    def update_chunk_content(self, chunk_id: str, new_content: str) -> bool:
        """更新指定分块的内容，重新嵌入并保存"""
        try:
            if not self.vectorstore or self.vectorstore_type != "faiss":
                return False

            idx = int(chunk_id)
            if idx not in self.vectorstore.index_to_docstore_id:
                return False

            docstore_id = self.vectorstore.index_to_docstore_id[idx]
            old_doc = self.vectorstore.docstore.search(docstore_id)
            if old_doc is None:
                return False

            meta = old_doc.metadata.copy()

            # Delete old vector by docstore UUID
            self.vectorstore.delete([docstore_id])

            # Generate new embedding
            embedding = self.embedding_generator.generate_embeddings([new_content])
            if not embedding or not embedding[0]:
                return False

            # Re-add with same metadata
            self.vectorstore.add_embeddings(
                [(new_content, embedding[0])],
                [meta],
            )
            self.vectorstore.save_local(self.persist_directory)
            logger.info(f"更新分块 {chunk_id} 成功")
            return True
        except Exception as e:
            logger.error(f"更新分块时出错: {e}")
            return False

    def delete_by_source(self, source: str) -> int:
        """删除指定源文件的所有分块，无需重建索引。返回删除的分块数。"""
        try:
            if not self.vectorstore:
                return 0

            if self.vectorstore_type == "faiss":
                docstore_ids: list[str] = []
                for idx, doc_id in list(self.vectorstore.index_to_docstore_id.items()):
                    doc = self.vectorstore.docstore.search(doc_id)
                    if doc is not None and doc.metadata.get("source") == source:
                        docstore_ids.append(doc_id)

                if not docstore_ids:
                    return 0

                self.vectorstore.delete(docstore_ids)
                self.vectorstore.save_local(self.persist_directory)
                logger.info(f"已从索引删除 {len(docstore_ids)} 个分块（源: {source}）")
                return len(docstore_ids)

            elif self.vectorstore_type == "chromadb":
                collection = self.vectorstore.get()
                ids_to_delete: list[str] = []
                ids_list = collection.get("ids", [])
                metas = collection.get("metadatas", [])
                for i, doc_id in enumerate(ids_list):
                    meta = metas[i] if metas and i < len(metas) else {}
                    if meta.get("source") == source:
                        ids_to_delete.append(str(doc_id))

                if ids_to_delete:
                    self.vectorstore.delete(ids=ids_to_delete)
                return len(ids_to_delete)

            return 0
        except Exception as e:
            logger.error(f"按源删除分块时出错: {e}")
            return 0

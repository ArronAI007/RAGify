from typing import Dict, Any, List, Optional, Union
from langchain_core.documents import Document
from .base import PipelineComponent
from ..core import (
    MultiModalDocumentLoader,
    MultiModalDocumentProcessor,
    MultiModalEmbeddingGenerator,
    VectorStoreManager
)
from ..config import get_config
import os


class DocumentLoaderComponent(PipelineComponent):
    """
    文档加载组件
    """
    
    def _setup(self) -> None:
        self.loader = MultiModalDocumentLoader()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行文档加载
        
        Args:
            data: 包含'file_paths'或'directory_path'的字典
            
        Returns:
            更新后的字典，添加'documents'字段
        """
        documents = []
        
        # 从文件路径加载
        if "file_paths" in data:
            for file_path in data["file_paths"]:
                if os.path.exists(file_path):
                    print(f"加载文件: {file_path}")
                    file_docs = self.loader.load_file(file_path)
                    documents.extend(file_docs)
        
        # 从目录加载
        elif "directory_path" in data:
            directory_path = data["directory_path"]
            if os.path.isdir(directory_path):
                print(f"加载目录: {directory_path}")
                glob_pattern = data.get("glob_pattern", "**/*")
                documents = self.loader.load_directory(directory_path, glob_pattern)
        
        # 从配置指定的默认数据目录加载
        else:
            config = get_config()
            data_dir = config.get("base.data_dir", "./data")
            if os.path.isdir(data_dir):
                print(f"从默认数据目录加载: {data_dir}")
                documents = self.loader.load_from_config()
        
        data["documents"] = documents
        data["document_count"] = len(documents)
        print(f"总共加载 {len(documents)} 个文档")
        
        return data
    
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 添加加载统计信息
        if "documents" in data:
            data["_loader_info"] = {
                "loaded_count": len(data["documents"]),
                "file_types": {}
            }
            
            # 统计文件类型
            for doc in data["documents"]:
                file_type = doc.metadata.get("file_type", "unknown")
                data["_loader_info"]["file_types"][file_type] = \
                    data["_loader_info"]["file_types"].get(file_type, 0) + 1
        
        return data


class DocumentProcessorComponent(PipelineComponent):
    """
    文档处理组件
    """
    
    def _setup(self) -> None:
        self.processor = MultiModalDocumentProcessor()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理文档
        
        Args:
            data: 包含'documents'的字典
            
        Returns:
            更新后的字典，添加'processed_documents'字段
        """
        if "documents" not in data or not data["documents"]:
            print("警告: 没有要处理的文档")
            data["processed_documents"] = []
            return data
        
        documents = data["documents"]
        print(f"处理 {len(documents)} 个文档")
        
        # 处理文档
        processed_docs = self.processor.process_multimodal_documents(documents)
        
        # 过滤短文档
        min_length = self.config.get("min_length", 20)
        filtered_docs = [doc for doc in processed_docs if len(doc.page_content.strip()) >= min_length]
        
        data["processed_documents"] = filtered_docs
        data["processed_document_count"] = len(filtered_docs)
        data["chunk_count"] = len(filtered_docs)
        
        print(f"处理完成，生成 {len(filtered_docs)} 个文档块")
        
        return data
    
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 添加处理统计信息
        if "processed_documents" in data:
            data["_processor_info"] = {
                "processed_count": len(data["processed_documents"]),
                "avg_chunk_size": sum(len(doc.page_content) for doc in data["processed_documents"]) / \
                    max(1, len(data["processed_documents"]))
            }
        
        return data


class EmbeddingGeneratorComponent(PipelineComponent):
    """
    嵌入生成组件
    """
    
    def _setup(self) -> None:
        self.embedding_generator = MultiModalEmbeddingGenerator()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成文档嵌入
        
        Args:
            data: 包含'processed_documents'的字典
            
        Returns:
            更新后的字典，添加'embeddings'字段
        """
        if "processed_documents" not in data or not data["processed_documents"]:
            print("警告: 没有要生成嵌入的文档")
            data["embeddings"] = []
            return data
        
        documents = data["processed_documents"]
        print(f"为 {len(documents)} 个文档生成嵌入向量")
        
        # 提取文本内容
        texts = [doc.page_content for doc in documents]
        
        # 生成嵌入
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # 过滤掉无法生成嵌入的文档和嵌入
        valid_docs_and_embeddings = []
        for doc, embedding in zip(documents, embeddings):
            if embedding:
                valid_docs_and_embeddings.append((doc, embedding))
        
        if valid_docs_and_embeddings:
            valid_docs, valid_embeddings = zip(*valid_docs_and_embeddings)
            data["documents_with_embeddings"] = list(valid_docs)
            data["embeddings"] = list(valid_embeddings)
            data["valid_document_count"] = len(valid_docs)
            print(f"成功生成 {len(valid_embeddings)} 个嵌入向量")
        else:
            data["documents_with_embeddings"] = []
            data["embeddings"] = []
            data["valid_document_count"] = 0
            print("警告: 无法生成任何嵌入向量")
        
        return data


class VectorStoreComponent(PipelineComponent):
    """
    向量存储组件
    """
    
    def _setup(self) -> None:
        self.vectorstore_manager = VectorStoreManager()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将文档添加到向量存储
        
        Args:
            data: 包含'documents_with_embeddings'或'processed_documents'的字典
            
        Returns:
            更新后的字典，添加'vectorstore_info'字段
        """
        # 确定要添加的文档
        documents_to_add = []
        if "documents_with_embeddings" in data:
            documents_to_add = data["documents_with_embeddings"]
        elif "processed_documents" in data:
            documents_to_add = data["processed_documents"]
        else:
            print("警告: 没有要添加到向量存储的文档")
            return data
        
        print(f"将 {len(documents_to_add)} 个文档添加到向量存储")
        
        # 添加到向量存储
        doc_ids = self.vectorstore_manager.add_documents(documents_to_add)
        
        # 获取向量存储信息
        total_docs = self.vectorstore_manager.get_document_count()
        
        data["vectorstore_info"] = {
            "added_count": len(doc_ids),
            "total_count": total_docs,
            "store_type": self.vectorstore_manager.vectorstore_type
        }
        
        print(f"成功添加 {len(doc_ids)} 个文档到向量存储，当前总文档数: {total_docs}")
        
        return data
    
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 检查是否需要清空向量存储
        if data.get("clear_vectorstore", False):
            print("清空向量存储")
            self.vectorstore_manager.clear()
        
        return data


class RetrieverComponent(PipelineComponent):
    """
    检索组件
    """
    
    def _setup(self) -> None:
        self.vectorstore_manager = VectorStoreManager()
        self.config = get_config()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行文档检索
        
        Args:
            data: 包含'query'的字典
            
        Returns:
            更新后的字典，添加'retrieved_documents'字段
        """
        if "query" not in data:
            print("警告: 检索组件需要查询参数")
            data["retrieved_documents"] = []
            return data
        
        query = data["query"]
        k = data.get("k", self.config.get("retrieval.k", 3))
        score_threshold = data.get("score_threshold", self.config.get("retrieval.score_threshold"))
        
        print(f"执行检索: '{query}'，k={k}")
        
        # 执行检索
        if score_threshold is not None:
            results_with_scores = self.vectorstore_manager.similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold
            )
            
            # 分离文档和分数
            retrieved_docs = []
            scores = []
            for doc, score in results_with_scores:
                retrieved_docs.append(doc)
                # 添加分数到文档元数据
                doc.metadata["retrieval_score"] = score
                scores.append(score)
            
            data["retrieved_documents"] = retrieved_docs
            data["retrieval_scores"] = scores
        else:
            # 不带分数的检索
            retrieved_docs = self.vectorstore_manager.similarity_search(
                query=query,
                k=k
            )
            data["retrieved_documents"] = retrieved_docs
        
        print(f"检索到 {len(data['retrieved_documents'])} 个相关文档")
        
        return data


class ResponseGeneratorComponent(PipelineComponent):
    """
    响应生成组件
    """
    
    def _setup(self) -> None:
        from ..core import LanguageModelManager
        self.language_model_manager = LanguageModelManager()
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成响应
        
        Args:
            data: 包含'query'和可选的'retrieved_documents'的字典
            
        Returns:
            更新后的字典，添加'response'字段
        """
        if "query" not in data:
            print("警告: 响应生成组件需要查询参数")
            return data
        
        query = data["query"]
        context_docs = data.get("retrieved_documents", [])
        
        print(f"生成响应，使用 {len(context_docs)} 个检索到的文档")
        
        # 创建RAG提示并生成响应
        if context_docs:
            prompt = self.language_model_manager.create_rag_prompt(query, context_docs)
        else:
            prompt = query
        
        response = self.language_model_manager.generate_response(prompt)
        
        data["response"] = response
        data["response_generated"] = True
        
        print("响应生成完成")
        
        return data

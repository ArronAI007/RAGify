from typing import Dict, Any, List, Optional
from .base import Pipeline
from .components import (
    DocumentLoaderComponent,
    DocumentProcessorComponent,
    EmbeddingGeneratorComponent,
    VectorStoreComponent,
    RetrieverComponent,
    ResponseGeneratorComponent
)
import logging


class IndexingPipeline(Pipeline):
    """
    RAG索引构建流水线
    用于加载、处理文档并构建向量索引
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化索引流水线
        
        Args:
            config: 流水线配置参数
        """
        super().__init__(config)
        
        # 添加组件到流水线
        self.add_component("document_loader", DocumentLoaderComponent())
        self.add_component("document_processor", DocumentProcessorComponent())
        self.add_component("embedding_generator", EmbeddingGeneratorComponent())
        self.add_component("vector_store", VectorStoreComponent())
        
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行索引流水线
        
        Args:
            data: 输入数据，应包含以下字段之一：
                - file_paths: 文件路径列表
                - directory_path: 目录路径
                - clear_vectorstore: 是否清空向量存储（可选，默认为False）
            
        Returns:
            流水线运行结果
        """
        logging.info("开始运行索引流水线")
        
        # 运行流水线
        result = super().run(data)
        
        # 生成索引摘要
        summary = self._generate_index_summary(result)
        result["indexing_summary"] = summary
        
        logging.info(f"索引流水线运行完成: {summary}")
        
        return result
    
    def _generate_index_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成索引构建摘要
        
        Args:
            result: 流水线运行结果
            
        Returns:
            索引摘要信息
        """
        return {
            "total_documents_loaded": result.get("document_count", 0),
            "total_documents_processed": result.get("processed_document_count", 0),
            "total_chunks_generated": result.get("chunk_count", 0),
            "total_documents_indexed": result.get("vectorstore_info", {}).get("added_count", 0),
            "vectorstore_total_docs": result.get("vectorstore_info", {}).get("total_count", 0),
            "vectorstore_type": result.get("vectorstore_info", {}).get("store_type", "unknown")
        }


class QueryPipeline(Pipeline):
    """
    RAG查询流水线
    用于处理用户查询、检索相关文档并生成响应
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化查询流水线
        
        Args:
            config: 流水线配置参数
        """
        super().__init__(config)
        
        # 添加组件到流水线
        self.add_component("retriever", RetrieverComponent())
        self.add_component("response_generator", ResponseGeneratorComponent())
        
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行查询流水线
        
        Args:
            data: 输入数据，应包含以下字段：
                - query: 用户查询文本
                - k: 检索文档数量（可选）
                - score_threshold: 检索分数阈值（可选）
            
        Returns:
            流水线运行结果，包含response字段
        """
        logging.info(f"开始运行查询流水线: {data.get('query', '').split('\n')[0][:50]}...")
        
        # 验证输入
        if "query" not in data or not data["query"]:
            raise ValueError("查询流水线需要 'query' 参数")
        
        # 运行流水线
        result = super().run(data)
        
        # 生成查询摘要
        summary = self._generate_query_summary(result)
        result["query_summary"] = summary
        
        logging.info("查询流水线运行完成")
        
        return result
    
    def _generate_query_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成查询处理摘要
        
        Args:
            result: 流水线运行结果
            
        Returns:
            查询摘要信息
        """
        retrieved_docs = result.get("retrieved_documents", [])
        scores = result.get("retrieval_scores", [])
        
        return {
            "query": result.get("query", ""),
            "documents_retrieved": len(retrieved_docs),
            "avg_retrieval_score": sum(scores) / max(1, len(scores)) if scores else 0,
            "response_generated": result.get("response_generated", False),
            "top_sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("retrieval_score", 0)
                }
                for doc in retrieved_docs[:3]  # 只显示前3个来源
            ]
        }


class MultiModalIndexingPipeline(IndexingPipeline):
    """
    多模态索引流水线
    支持处理包含图像、音频等多模态内容的文档
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多模态索引流水线
        
        Args:
            config: 流水线配置参数
        """
        super().__init__(config)
        
        # 可以在这里添加多模态特定的配置或组件
        self.config.setdefault("multimodal_enabled", True)
    
    def _generate_index_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成多模态索引摘要
        
        Args:
            result: 流水线运行结果
            
        Returns:
            索引摘要信息，包含多模态内容统计
        """
        # 调用父类的摘要生成方法
        summary = super()._generate_index_summary(result)
        
        # 添加多模态特定信息
        if "_loader_info" in result:
            file_types = result["_loader_info"].get("file_types", {})
            # 计算多模态文件的比例
            image_types = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg"]
            audio_types = ["mp3", "wav", "ogg", "flac"]
            video_types = ["mp4", "avi", "mov", "wmv", "flv"]
            
            multimodal_count = 0
            for file_type, count in file_types.items():
                file_type_lower = file_type.lower()
                if (file_type_lower in image_types or 
                    file_type_lower in audio_types or 
                    file_type_lower in video_types):
                    multimodal_count += count
            
            # 添加到摘要
            total_files = sum(file_types.values())
            if total_files > 0:
                summary["multimodal_files_count"] = multimodal_count
                summary["multimodal_files_percentage"] = (multimodal_count / total_files) * 100
        
        return summary


class MultiModalQueryPipeline(QueryPipeline):
    """
    多模态查询流水线
    支持处理包含图像、音频等多模态内容的查询
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多模态查询流水线
        
        Args:
            config: 流水线配置参数
        """
        super().__init__(config)
        
        # 可以在这里添加多模态特定的配置或组件
        self.config.setdefault("multimodal_enabled", True)
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行多模态查询流水线
        
        Args:
            data: 输入数据，除了标准查询参数外，还可以包含：
                - image_urls: 图像URL列表
                - audio_urls: 音频URL列表
            
        Returns:
            流水线运行结果
        """
        # 检查是否有图像查询
        has_multimodal_input = any(key in data for key in ["image_urls", "audio_urls"])
        
        if has_multimodal_input:
            logging.info("处理多模态查询")
            # 在这个示例中，我们只是记录信息
            # 在实际实现中，可以添加特殊处理逻辑
        
        # 运行标准查询流水线
        return super().run(data)


class MultiStagePipeline(Pipeline):
    """
    多级流水线
    组合多个子流水线，支持复杂的多步骤处理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多级流水线
        
        Args:
            config: 流水线配置参数
        """
        super().__init__(config)
        
        # 创建子流水线
        self.indexing_pipeline = IndexingPipeline(config)
        self.query_pipeline = QueryPipeline(config)
        
    def run_indexing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行索引子流水线
        
        Args:
            data: 索引流水线输入数据
            
        Returns:
            索引流水线结果
        """
        return self.indexing_pipeline.run(data)
    
    def run_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行查询子流水线
        
        Args:
            data: 查询流水线输入数据
            
        Returns:
            查询流水线结果
        """
        return self.query_pipeline.run(data)
    
    def run_complete_workflow(self, index_data: Dict[str, Any], query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行完整的工作流：先索引，后查询
        
        Args:
            index_data: 索引流水线输入数据
            query_data: 查询流水线输入数据
            
        Returns:
            包含索引和查询结果的字典
        """
        # 运行索引流水线
        index_result = self.run_indexing(index_data)
        
        # 运行查询流水线
        query_result = self.run_query(query_data)
        
        # 合并结果
        result = {
            "indexing_result": index_result,
            "query_result": query_result
        }
        
        return result


# 工厂函数

def create_pipeline(pipeline_type: str, config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    创建流水线的工厂函数
    
    Args:
        pipeline_type: 流水线类型
            - "indexing": 标准索引流水线
            - "query": 标准查询流水线
            - "multimodal_indexing": 多模态索引流水线
            - "multimodal_query": 多模态查询流水线
            - "multi_stage": 多级流水线
        config: 流水线配置
        
    Returns:
        创建的流水线实例
        
    Raises:
        ValueError: 如果流水线类型无效
    """
    pipeline_classes = {
        "indexing": IndexingPipeline,
        "query": QueryPipeline,
        "multimodal_indexing": MultiModalIndexingPipeline,
        "multimodal_query": MultiModalQueryPipeline,
        "multi_stage": MultiStagePipeline
    }
    
    if pipeline_type not in pipeline_classes:
        raise ValueError(f"无效的流水线类型: {pipeline_type}. 支持的类型: {list(pipeline_classes.keys())}")
    
    pipeline_class = pipeline_classes[pipeline_type]
    return pipeline_class(config)

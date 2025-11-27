"""
MCP (Multi-Component Pipeline) 模块
提供灵活的RAG流水线组件和配置
"""

# 基础组件
from .base import PipelineComponent, Pipeline

# 具体组件实现
from .components import (
    DocumentLoaderComponent,
    DocumentProcessorComponent,
    EmbeddingGeneratorComponent,
    VectorStoreComponent,
    RetrieverComponent,
    ResponseGeneratorComponent
)

# 预定义流水线
from .pipelines import (
    IndexingPipeline,
    QueryPipeline,
    MultiModalIndexingPipeline,
    MultiModalQueryPipeline,
    MultiStagePipeline,
    create_pipeline
)

__all__ = [
    # 基础类
    "PipelineComponent",
    "Pipeline",
    
    # 组件
    "DocumentLoaderComponent",
    "DocumentProcessorComponent",
    "EmbeddingGeneratorComponent",
    "VectorStoreComponent",
    "RetrieverComponent",
    "ResponseGeneratorComponent",
    
    # 流水线
    "IndexingPipeline",
    "QueryPipeline",
    "MultiModalIndexingPipeline",
    "MultiModalQueryPipeline",
    "MultiStagePipeline",
    
    # 工厂函数
    "create_pipeline"
]
"""
RAGify - 多模态RAG框架
"""

__version__ = "0.1.0"

# 导出核心模块
from .config import get_config, ConfigLoader, RAGifySettings
from .core import (
    MultiModalDocumentLoader, ImageDocumentProcessor,
    DocumentProcessor, MultiModalDocumentProcessor,
    EmbeddingGenerator, MultiModalEmbeddingGenerator,
    VectorStoreManager,
    LanguageModelManager, MultiModalLanguageModelManager,
    RAGChainManager, MultiModalRAGChainManager
)
from .mcp import (
    PipelineComponent, Pipeline,
    DocumentLoaderComponent, DocumentProcessorComponent,
    EmbeddingGeneratorComponent, VectorStoreComponent,
    RetrieverComponent, ResponseGeneratorComponent,
    IndexingPipeline, QueryPipeline,
    MultiModalIndexingPipeline, MultiModalQueryPipeline,
    MultiStagePipeline, create_pipeline
)
# 暂时注释掉agents模块的导入
# from .agents import (
#     RAGifyAgent, RAGifyTool, AgentRegistry,
#     RAGAgent, MultiModalRAGAgent, PipelineAgent,
#     get_default_tools, create_custom_tool,
#     agent_registry
# )
from .cli import ragify

__all__ = [
    # 配置相关
    'get_config', 'ConfigLoader', 'RAGifySettings',
    # 核心组件
    'MultiModalDocumentLoader', 'ImageDocumentProcessor',
    'DocumentProcessor', 'MultiModalDocumentProcessor',
    'EmbeddingGenerator', 'MultiModalEmbeddingGenerator',
    'VectorStoreManager',
    'LanguageModelManager', 'MultiModalLanguageModelManager',
    'RAGChainManager', 'MultiModalRAGChainManager',
    # MCP相关
    'PipelineComponent', 'Pipeline',
    'DocumentLoaderComponent', 'DocumentProcessorComponent',
    'EmbeddingGeneratorComponent', 'VectorStoreComponent',
    'RetrieverComponent', 'ResponseGeneratorComponent',
    'IndexingPipeline', 'QueryPipeline',
    'MultiModalIndexingPipeline', 'MultiModalQueryPipeline',
    'MultiStagePipeline', 'create_pipeline',
    # Agent相关
    'RAGifyAgent', 'RAGifyTool', 'AgentRegistry',
    'RAGAgent', 'MultiModalRAGAgent', 'PipelineAgent',
    'get_default_tools', 'create_custom_tool',
    'agent_registry',
    # CLI
    'ragify'
]
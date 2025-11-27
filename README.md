# RAGify

一个强大、灵活的多模态RAG（检索增强生成）框架，支持MCP（多组件流水线）和Agent功能，基于LangChain 1.0构建。

## 功能特性

### 🔍 核心RAG功能
- 支持多种文档格式（文本、PDF、Word、图像等）
- 灵活的文档分块策略
- 多种嵌入模型支持（OpenAI、HuggingFace等）
- 多种向量数据库支持（Chroma、FAISS等）
- 高效的相似度检索

### 🖼️ 多模态支持
- 图像内容处理与OCR文本提取
- 多模态嵌入与检索
- 混合内容文档处理
- 跨模态查询能力

### 🔄 MCP（多组件流水线）
- 模块化的组件设计
- 可自定义的流水线配置
- 内置多种预定义流水线
- 灵活的组件组合与扩展

### 🤖 Agent框架
- 多种Agent类型（RAGAgent、MultiModalRAGAgent、PipelineAgent）
- 丰富的内置工具集
- Agent注册与管理
- 支持工具调用和会话管理

### ⚙️ 灵活配置
- YAML格式配置文件
- 环境变量支持
- 运行时配置覆盖
- 组件级配置自定义

## 项目结构

```
RAGify/
├── config/                    # 配置文件目录
│   └── config.yaml            # 默认配置文件
├── ragify/                    # 主源码目录
│   ├── __init__.py
│   ├── config/                # 配置管理模块
│   ├── core/                  # 核心RAG功能模块
│   ├── mcp/                   # 多组件流水线模块
│   └── agents/                # Agent模块
├── examples/                  # 示例脚本目录
├── tests/                     # 测试目录
├── pyproject.toml             # Python包配置
└── README.md                  # 项目文档
```

## 快速开始

### 1. 安装依赖

RAGify使用`uv`进行依赖管理，确保你已经安装了uv：

```bash
# 安装uv（如果尚未安装）
pip install uv

# 使用uv安装项目依赖
uv venv
uv pip install -e .
```

### 2. 配置环境

创建并编辑配置文件：

```bash
cp config/config.yaml.example config/config.yaml
```

根据你的环境修改配置文件，主要配置项包括：

- LLM配置（OpenAI API密钥等）
- 嵌入模型配置
- 向量数据库配置
- 多模态处理配置

### 3. 运行示例

```bash
# 基础RAG示例
python examples/basic_rag_example.py

# 多模态RAG示例
python examples/multimodal_rag_example.py

# Agent示例
python examples/agent_example.py
```

## 使用指南

### 基础RAG用法

```python
from ragify.mcp import IndexingPipeline, QueryPipeline

# 1. 索引文档
index_pipeline = IndexingPipeline()
index_result = index_pipeline.run({
    "directory_path": "./your_documents",
    "clear_vectorstore": True
})

# 2. 执行查询
query_pipeline = QueryPipeline()
result = query_pipeline.run({"query": "你的问题"})
print(result["response"])
```

### 使用Agent

```python
from ragify.agents import RAGAgent, get_default_tools

# 创建带工具的RAGAgent
agent = RAGAgent(tools=get_default_tools())

# 提问
response = agent.ask("你的问题")
print(response)
```

### 多模态处理

```python
from ragify.mcp import MultiModalIndexingPipeline, MultiModalQueryPipeline

# 索引多模态内容
mm_indexer = MultiModalIndexingPipeline()
mm_indexer.run({"directory_path": "./multimodal_documents"})

# 执行多模态查询
mm_query = MultiModalQueryPipeline()
result = mm_query.run({"query": "描述图像中的内容"})
```

## 配置详解

配置文件采用YAML格式，主要配置项包括：

### 基本配置

```yaml
basic:
  project_name: "RAGify"
  log_level: "INFO"
  debug: false
```

### LLM配置

```yaml
llm:
  provider: "openai"  # 可选: openai, anthropic
  model: "gpt-4o"
  temperature: 0.7
  max_tokens: 2048
  api_key_env: "OPENAI_API_KEY"  # 环境变量名
```

### 嵌入模型配置

```yaml
embeddings:
  provider: "openai"  # 可选: openai, huggingface
  model: "text-embedding-3-small"
  dimensions: 1536
  api_key_env: "OPENAI_API_KEY"
```

### 向量数据库配置

```yaml
vectorstore:
  type: "chromadb"  # 可选: chromadb, faiss
  persist_directory: "./vectorstore"
  collection_name: "default"
```

### 多模态配置

```yaml
multimodal:
  enabled: true
  ocr_enabled: true
  image_processing:
    enabled: true
    max_size: 1024
```

## 高级功能

### 自定义组件

你可以继承基础组件类来创建自定义组件：

```python
from ragify.mcp import PipelineComponent

class MyCustomComponent(PipelineComponent):
    def __init__(self, config=None):
        super().__init__(config)
    
    def process(self, data):
        # 实现自定义处理逻辑
        processed_data = {"processed": "your custom processing"}
        return processed_data
```

### 创建自定义Agent

```python
from ragify.agents import RAGifyAgent, agent_registry

@agent_registry.register("my_custom_agent")
class MyCustomAgent(RAGifyAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def ask(self, query):
        # 自定义提问逻辑
        return "Custom response to: " + query
```

### 创建自定义工具

```python
from ragify.agents import RAGifyTool

def my_custom_tool_function(param1, param2):
    """自定义工具函数的文档字符串"""
    return f"Result of {param1} and {param2}"

# 创建自定义工具
my_tool = RAGifyTool(
    name="my_custom_tool",
    func=my_custom_tool_function,
    description="这是一个自定义工具",
    params_schema={
        "param1": {"type": "string", "description": "第一个参数"},
        "param2": {"type": "string", "description": "第二个参数"}
    }
)
```

## 依赖项

- Python 3.10+
- LangChain 1.0+
- OpenAI SDK (可选，用于OpenAI模型)
- Anthropic SDK (可选，用于Claude模型)
- ChromaDB
- FAISS
- Pillow (用于图像处理)
- pytesseract (用于OCR)
- PyYAML (用于配置文件)
- Pydantic (用于数据验证)

## 开发指南

### 安装开发依赖

```bash
uv pip install -e "[dev]"
```

### 运行测试

```bash
pytest tests/
```

### 代码风格

项目使用black和isort进行代码格式化：

```bash
black ragify/
examples/
tests/
isort ragify/
examples/
tests/
```

## 许可证

MIT License

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request

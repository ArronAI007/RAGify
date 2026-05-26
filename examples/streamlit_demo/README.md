# RAGify Streamlit Demo

基于 RAGify 框架的企业级智能知识库演示页面。

## 功能展示

- 📤 **文档索引** - 支持 PDF/Word/HTML/图片多种格式
- 💬 **智能问答** - 基于向量检索的 RAG 回答
- 🖼️ **多模态处理** - 图片内容理解与跨模态查询
- 🏢 **企业级特性** - 私有化部署、灵活扩展、快速交付

## 快速启动

```bash
# 进入项目目录
cd /path/to/RAGify

# 方式一：使用 uv（推荐）
uv pip install streamlit
uv pip install -e .
streamlit run examples/streamlit_demo/app.py

# 方式二：使用 pip
pip install streamlit
pip install -e .
streamlit run examples/streamlit_demo/app.py
```

## 环境变量

确保在环境变量或 `.env` 文件中配置：

```bash
OPENAI_API_KEY=your_api_key_here
```

## 界面预览

启动后访问 `http://localhost:8501` 即可看到演示页面。

## ⚠️ 注意事项

- 这是演示版本，生产环境请使用完整部署方案
- 向量数据库默认使用 Chroma（内存模式）
- 如需持久化存储，请修改 `config/config.yaml`

## License

MIT License - ArronAI007/RAGify
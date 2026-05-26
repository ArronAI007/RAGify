"""
RAGify Streamlit Demo
快速演示RAGify核心功能的企业级Demo页面
使用方法: streamlit run app.py
"""

import streamlit as st
import os
import sys
from pathlib import Path

# 导入RAGify核心模块
from ragify.mcp import IndexingPipeline, QueryPipeline, MultiModalIndexingPipeline, MultiModalQueryPipeline
from ragify.agents import RAGAgent, get_default_tools

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAGify - 企业智能知识库",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 样式 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E3A5F;
    }
    .success-box {
        background: #d4edda;
        border-radius: 8px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.image("https://img.shields.io/badge/RAGify-v1.0-blue", height=30)
    st.title("⚙️ 配置")
    
    # LLM配置
    st.subheader("🤖 LLM配置")
    provider = st.selectbox("模型提供商", ["openai", "anthropic"], help="选择LLM服务商")
    model = st.selectbox("模型", ["gpt-4o", "gpt-4o-mini", "claude-3-sonnet"], help="选择具体模型")
    
    # 向量库配置
    st.subheader("📦 向量数据库")
    vectorstore_type = st.selectbox("向量库", ["chromadb", "faiss"], help="选择向量数据库")
    
    # 嵌入模型
    st.subheader("🔤 嵌入模型")
    embed_provider = st.selectbox("嵌入提供商", ["openai", "huggingface"], help="选择嵌入模型")
    embed_model = st.selectbox("嵌入模型", ["text-embedding-3-small", "text-embedding-3", "bge-large"], help="选择嵌入模型")
    
    st.divider()
    st.caption("RAGify - 让RAG开发像拼积木一样简单")

# ==================== 主内容区 ====================
st.markdown('<p class="main-header">🔍 RAGify 企业智能知识库</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于自研RAG框架，支持多模态文档理解与智能问答</p>', unsafe_allow_html=True)

# ==================== 功能展示区 ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 支持格式", "10+", "PDF/Word/HTML/图片")
with col2:
    st.metric("🔤 嵌入模型", "5+", "OpenAI/BGE/HuggingFace")
with col3:
    st.metric("📦 向量库", "3+", "Chroma/FAISS/Milvus")
with col4:
    st.metric("🤖 Agent", "内置", "多类型支持")

st.divider()

# ==================== 核心功能Tab ====================
tab1, tab2, tab3 = st.tabs(["📤 文档索引", "💬 智能问答", "🖼️ 多模态"])

# Tab 1: 文档索引
with tab1:
    st.header("文档索引")
    st.write("上传你的文档，自动构建向量知识库")
    
    col_upload, col_config = st.columns([2, 1])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "拖拽文件或点击上传", 
            type=["pdf", "docx", "txt", "md", "html"],
            accept_multiple_files=True,
            help="支持多种文档格式"
        )
    
    with col_config:
        chunk_size = st.slider("分块大小", 256, 2048, 512, help="文档分块大小")
        overlap = st.slider("重叠 token", 0, 256, 64, help="相邻块重叠量")
        clear_before = st.checkbox("清空已有索引", value=False)
    
    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件")
        
        if st.button("🚀 开始索引", type="primary", use_container_width=True):
            with st.spinner("索引中..."):
                # 保存上传文件
                temp_dir = Path("./temp_documents")
                temp_dir.mkdir(exist_ok=True)
                
                for file in uploaded_files:
                    (temp_dir / file.name).write_bytes(file.getvalue())
                
                # 执行索引
                index_pipeline = IndexingPipeline()
                result = index_pipeline.run({
                    "directory_path": str(temp_dir),
                    "clear_vectorstore": clear_before
                })
                
                st.success(f"索引完成！处理了 {result.get('document_count', '?')} 个文档")
                
                # 清理临时文件
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

# Tab 2: 智能问答
with tab2:
    st.header("智能问答")
    st.write("基于知识库进行智能问答，支持引用来源")
    
    query = st.text_input("💬 请输入你的问题", placeholder="例如：公司的年假政策是什么？")
    top_k = st.slider("返回结果数", 1, 10, 3)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    col_query, col_result = st.columns([1, 2])
    
    with col_query:
        if st.button("🔍 查询", type="primary", use_container_width=True):
            if not query:
                st.warning("请先输入问题")
            else:
                with st.spinner("思考中..."):
                    query_pipeline = QueryPipeline()
                    result = query_pipeline.run({
                        "query": query,
                        "top_k": top_k,
                        "temperature": temperature
                    })
                    
                    with col_result:
                        st.success("回答:")
                        st.markdown(result.get("response", "无回答"))
                        
                        # 显示来源
                        if "sources" in result and result["sources"]:
                            st.divider()
                            st.subheader("📎 参考来源")
                            for i, src in enumerate(result["sources"][:3]):
                                with st.container():
                                    st.caption(f"来源 {i+1}: {src.get('source', 'unknown')}")
                                    st.text(src.get("content", "")[:200] + "...")

# Tab 3: 多模态
with tab3:
    st.header("多模态处理")
    st.write("支持图片内容理解与跨模态查询")
    
    col_img, col_img_query = st.columns([1, 1])
    
    with col_img:
        image_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "gif"])
        if image_file:
            st.image(image_file, use_container_width=True)
    
    with col_img_query:
        if image_file:
            img_query = st.text_input("🖼️ 图片相关问题", placeholder="描述这张图片中的内容")
            if st.button("🖼️ 分析图片", type="primary") and img_query:
                with st.spinner("分析中..."):
                    mm_query = MultiModalQueryPipeline()
                    result = mm_query.run({"query": img_query})
                    st.success("分析结果:")
                    st.markdown(result.get("response", "无结果"))

st.divider()

# ==================== 企业特性 ====================
st.subheader("🏢 企业级特性")

col_feat1, col_feat2, col_feat3 = st.columns(3)

with col_feat1:
    st.markdown("""
    <div class="feature-card">
    <h4>🔒 私有化部署</h4>
    <p>数据不出企业网络，支持本地化部署，满足数据安全合规要求</p>
    </div>
    """, unsafe_allow_html=True)

with col_feat2:
    st.markdown("""
    <div class="feature-card">
    <h4>🔧 灵活扩展</h4>
    <p>模块化架构，支持自定义组件，满足各类复杂业务场景</p>
    </div>
    """, unsafe_allow_html=True)

with col_feat3:
    st.markdown("""
    <div class="feature-card">
    <h4>⚡ 快速交付</h4>
    <p>基于成熟框架，最快1周完成企业知识库搭建</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== 定价 ====================
st.divider()
st.subheader("💰 服务方案")

col_plan1, col_plan2, col_plan3 = st.columns(3)

with col_plan1:
    st.markdown("""
    <div class="feature-card">
    <h3>🌀 体验版</h3>
    <h2>¥8,000</h2>
    <ul>
        <li>原型验证</li>
        <li>≤5份文档</li>
        <li>基础问答</li>
        <li>7×24h 支持</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_plan2:
    st.markdown("""
    <div style="background:#1E3A5F;color:white;border-radius:10px;padding:1.5rem;margin:0.5rem 0;">
    <h3>🏢 企业版</h3>
    <h2>¥15,000</h2>
    <ul>
        <li>完整RAG系统</li>
        <li>文档不限量</li>
        <li>Agent能力</li>
        <li>多模态支持</li>
        <li>技术培训</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_plan3:
    st.markdown("""
    <div class="feature-card">
    <h3>🚀 旗舰版</h3>
    <h2>¥30,000+</h2>
    <ul>
        <li>私有化全套部署</li>
        <li>多租户支持</li>
        <li>定制开发</li>
        <li>年度运维</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== 页脚 ====================
st.divider()
st.markdown("""
<div style="text-align:center;color:#666;font-size:0.8rem;">
<p>RAGify © 2024-2026 | 基于LangChain 1.0构建 | MIT开源协议</p>
<p>📧 联系我们 | 📱 扫码咨询</p>
</div>
""", unsafe_allow_html=True)
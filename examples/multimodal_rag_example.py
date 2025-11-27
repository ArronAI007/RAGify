#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG示例
演示如何使用RAGify处理包含图像的多模态文档
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragify.mcp import MultiModalIndexingPipeline, MultiModalQueryPipeline
from ragify.config import get_config


def main():
    """
    主函数，演示多模态RAG流程
    """
    print("===== RAGify 多模态示例 =====")
    
    # 获取配置
    config = get_config()
    print(f"使用配置文件: {config.config_path}")
    
    # 创建示例文档（如果不存在）
    create_sample_multimodal_documents()
    
    # 1. 索引多模态文档
    print("\n1. 开始索引多模态文档...")
    index_pipeline = MultiModalIndexingPipeline()
    sample_data_dir = os.path.join(os.path.dirname(__file__), "multimodal_data")
    
    index_result = index_pipeline.run({
        "directory_path": sample_data_dir,
        "clear_vectorstore": True  # 每次运行都清空向量存储
    })
    
    # 显示索引结果
    summary = index_result.get("indexing_summary", {})
    print("\n索引完成!")
    print(f"- 加载的文档总数: {summary.get('total_documents_loaded', 0)}")
    print(f"- 处理的文档总数: {summary.get('total_documents_processed', 0)}")
    print(f"- 处理的图像总数: {summary.get('total_images_processed', 0)}")
    print(f"- 生成的嵌入向量总数: {summary.get('total_embeddings_generated', 0)}")
    print(f"- 索引的内容总数: {summary.get('total_contents_indexed', 0)}")
    print(f"- 向量存储类型: {summary.get('vectorstore_type', 'unknown')}")
    
    # 2. 执行多模态查询
    print("\n2. 开始多模态查询示例...")
    query_pipeline = MultiModalQueryPipeline()
    
    # 示例查询
    text_queries = [
        "图像中显示的是什么内容？",
        "多模态RAG如何处理图像信息？",
        "有关于图表分析的示例吗？"
    ]
    
    for i, query in enumerate(text_queries, 1):
        print(f"\n文本查询 {i}: {query}")
        print("-" * 60)
        
        # 执行查询
        query_result = query_pipeline.run({"query": query})
        
        # 显示结果
        response = query_result.get("response", "未生成响应")
        print(f"响应: {response}")
        
        # 显示检索到的文档信息
        retrieved_contents = query_result.get("retrieved_contents", [])
        print(f"\n检索到 {len(retrieved_contents)} 个相关内容:")
        for j, content in enumerate(retrieved_contents[:3], 1):  # 只显示前3个
            source = content.metadata.get("source", "未知来源")
            content_type = content.metadata.get("content_type", "unknown")
            score = content.metadata.get("retrieval_score", "未知")
            
            print(f"  来源 {j}: {source} (类型: {content_type}, 相关度: {score})")
            
            # 根据内容类型显示预览
            if content_type == "text":
                preview = content.page_content[:100] + "..." if len(content.page_content) > 100 else content.page_content
                print(f"  内容预览: {preview}")
            elif content_type == "image":
                image_path = content.metadata.get("image_path", "")
                print(f"  图像路径: {image_path}")
                print(f"  OCR文本: {content.page_content[:150]}..." if len(content.page_content) > 150 else f"  OCR文本: {content.page_content}")
    
    print("\n===== 多模态示例完成 =====")
    print("\n注意：本示例使用文本描述图像内容，实际多模态RAG系统可以处理真实图像文件。")


def create_sample_multimodal_documents():
    """
    创建示例多模态文档用于演示
    """
    multimodal_data_dir = os.path.join(os.path.dirname(__file__), "multimodal_data")
    os.makedirs(multimodal_data_dir, exist_ok=True)
    
    # 创建示例文本描述文件（模拟图像内容）
    img_desc_path = os.path.join(multimodal_data_dir, "image_descriptions.txt")
    if not os.path.exists(img_desc_path):
        with open(img_desc_path, "w", encoding="utf-8") as f:
            f.write("""
# 图像内容描述（模拟OCR结果）

## 图像1: 多模态RAG架构图
[图像路径: sample_architecture_diagram.png]
[图像类型: diagram]
[OCR文本]:

多模态RAG系统架构

1. 文档输入层
   - 文本文档
   - 图像文档
   - PDF文档（含图表）
   - 混合内容文档

2. 处理层
   - 多模态文档加载器
   - 文本分块器
   - 图像预处理
   - OCR提取
   - 多模态嵌入生成

3. 存储层
   - 向量数据库
   - 元数据存储
   - 图像索引

4. 检索层
   - 跨模态检索
   - 混合检索
   - 相关度排序

5. 生成层
   - 多模态上下文增强
   - 回答生成
   - 多模态输出

## 图像2: 数据流程示意图
[图像路径: data_flow_diagram.png]
[图像类型: flow_chart]
[OCR文本]:

多模态RAG数据流程

查询输入 -> 模态检测 -> 查询处理 -> 混合检索 -> 上下文构建 -> 模型生成 -> 响应输出

处理步骤详情:
1. 接收用户查询（文本/图像）
2. 检测查询模态类型
3. 将查询转换为多模态嵌入
4. 在向量库中检索相关内容
5. 构建包含文本和图像的混合上下文
6. 输入多模态LLM生成回答
7. 输出格式化的响应

## 图像3: 性能对比图表
[图像路径: performance_comparison.png]
[图像类型: chart]
[OCR文本]:

RAG系统性能对比

| 系统类型 | 准确率 | 响应时间 | 资源消耗 |
|---------|-------|---------|--------|
| 传统RAG | 85%   | 1.2s    | 中    |
| 多模态RAG | 92%  | 2.5s    | 高    |
| 优化多模态RAG | 94% | 1.8s    | 中高  |

关键指标：
- 准确率提升：多模态RAG比传统RAG高7%
- 响应时间：多模态处理需要更多计算
- 资源消耗：可通过优化算法降低
            """.strip())
    
    # 创建多模态处理指南
    guide_path = os.path.join(multimodal_data_dir, "multimodal_processing_guide.txt")
    if not os.path.exists(guide_path):
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write("""
# 多模态内容处理指南

## 图像预处理步骤

1. **图像加载与缩放**
   - 加载图像文件
   - 调整到标准尺寸
   - 确保图像质量

2. **OCR文本提取**
   - 使用Tesseract或其他OCR引擎
   - 文本区域检测
   - 文本识别与提取
   - 后处理与清理

3. **图像特征提取**
   - 使用CNN模型提取视觉特征
   - 生成图像嵌入向量
   - 提取图像元数据

## 混合文档处理策略

1. **PDF文档处理**
   - 提取文本内容
   - 提取嵌入图像
   - 保留页面结构信息
   - 处理表格数据

2. **Word文档处理**
   - 提取文本段落
   - 处理格式信息
   - 提取图像和图表
   - 保留文档结构

3. **PowerPoint处理**
   - 提取幻灯片内容
   - 处理文本和图像
   - 保留演示逻辑

## 多模态嵌入技术

1. **文本嵌入**
   - 使用CLIP或其他多模态模型
   - 将文本映射到共享向量空间

2. **图像嵌入**
   - 使用视觉编码器
   - 生成图像的向量表示
   - 确保与文本在同一向量空间

3. **嵌入融合策略**
   - 早期融合：在嵌入前合并特征
   - 晚期融合：在检索后合并结果
   - 混合融合：结合多种融合方法

## 跨模态检索优化

1. **检索策略**
   - 文本到图像检索
   - 图像到文本检索
   - 文本到混合内容检索

2. **相关度计算**
   - 向量相似度计算
   - 跨模态相似度调整
   - 混合相似度排序

3. **结果优化**
   - 多模态结果过滤
   - 重排序技术
   - 上下文感知检索
            """.strip())
    
    # 创建示例图表分析文档
    chart_path = os.path.join(multimodal_data_dir, "chart_analysis_examples.txt")
    if not os.path.exists(chart_path):
        with open(chart_path, "w", encoding="utf-8") as f:
            f.write("""
# 图表分析示例

## 图表类型识别与处理

1. **柱状图分析**
   - 识别柱状图结构
   - 提取X轴和Y轴标签
   - 识别数据值
   - 分析比较关系

2. **折线图分析**
   - 识别趋势线
   - 提取时间序列数据
   - 分析趋势变化
   - 识别关键转折点

3. **饼图分析**
   - 识别扇区分布
   - 提取比例信息
   - 分析占比关系

## 图表OCR与理解

1. **表格数据提取**
   - 使用表格识别算法
   - 提取表头信息
   - 识别单元格内容
   - 重建表格结构

2. **数学公式识别**
   - 使用公式OCR引擎
   - 识别数学符号
   - 重建公式结构
   - 生成可计算表示

3. **科学图表理解**
   - 识别坐标轴和刻度
   - 提取实验数据点
   - 分析科学关系
   - 理解实验结论

## 多模态RAG中的图表处理流程

1. **图表文档处理**
   - 提取图表图像
   - 执行OCR提取
   - 图表类型分类
   - 结构化信息提取

2. **图表向量化**
   - 提取视觉特征
   - 结合文本描述
   - 生成图表嵌入
   - 存储到向量数据库

3. **查询与检索**
   - 理解用户对图表的查询
   - 检索相关图表
   - 提取图表信息
   - 生成包含图表分析的回答
            """.strip())
    
    print(f"示例多模态文档已创建在: {multimodal_data_dir}")


if __name__ == "__main__":
    main()

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from ..config import get_config
import re


class DocumentProcessor:
    """
    文档处理器，负责文本分块、清理和预处理
    """
    
    def __init__(self):
        self.config = get_config()
        self.chunk_size = self.config.get("retrieval.chunk_size", 1000)
        self.chunk_overlap = self.config.get("retrieval.chunk_overlap", 100)
    
    def clean_text(self, text: str) -> str:
        """
        清理文本，移除多余的空白字符和换行符
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 保留段落结构的同时清理多余换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 去除首尾空白
        text = text.strip()
        return text
    
    def split_text(self, text: str, file_type: str = ".txt") -> List[str]:
        """
        根据文件类型选择合适的文本分块器
        """
        # 根据文件类型选择合适的分块器
        if file_type == ".md":
            text_splitter = MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif file_type in [".txt", ".pdf", ".docx", ".doc"]:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", "", ". ", "! ", "? "]
            )
        else:
            # 默认使用递归字符分块器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        
        return text_splitter.split_text(text)
    
    def process_document(self, document: Document) -> List[Document]:
        """
        处理单个文档，返回分块后的文档列表
        """
        # 清理文本
        cleaned_text = self.clean_text(document.page_content)
        
        # 获取文件类型
        file_type = document.metadata.get("file_type", ".txt")
        
        # 分块文本
        chunks = self.split_text(cleaned_text, file_type)
        
        # 创建分块文档
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            # 复制原始元数据并添加分块信息
            metadata = document.metadata.copy()
            metadata["chunk_index"] = i
            metadata["chunk_total"] = len(chunks)
            metadata["chunk_size"] = len(chunk)
            
            chunk_doc = Document(page_content=chunk, metadata=metadata)
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        处理多个文档
        """
        processed_docs = []
        
        for doc in documents:
            chunk_docs = self.process_document(doc)
            processed_docs.extend(chunk_docs)
        
        return processed_docs
    
    def filter_documents(self, documents: List[Document], min_length: int = 20) -> List[Document]:
        """
        过滤过短的文档
        """
        return [doc for doc in documents if len(doc.page_content.strip()) >= min_length]


class MultiModalDocumentProcessor:
    """
    多模态文档处理器，处理包含文本和图像的混合内容
    """
    
    def __init__(self):
        self.config = get_config()
        self.document_processor = DocumentProcessor()
        self.multimodal_enabled = self.config.get("multimodal.enabled", True)
    
    def is_multimodal(self, document: Document) -> bool:
        """
        判断文档是否为多模态文档
        """
        file_type = document.metadata.get("file_type", "").lower()
        return file_type in [".jpg", ".jpeg", ".png", ".gif"] or \
               "image" in document.metadata
    
    def process_multimodal_document(self, document: Document) -> List[Document]:
        """
        处理多模态文档
        """
        if not self.is_multimodal(document):
            # 非多模态文档，使用常规处理器
            return self.document_processor.process_document(document)
        
        # 对于图像文档，我们可能需要特殊处理
        # 这里简单地将图像信息作为内容处理
        # 在实际应用中，可能需要提取图像特征或使用OCR
        
        # 添加多模态标记
        document.metadata["is_multimodal"] = True
        
        # 如果有OCR文本，可以在这里处理
        ocr_text = document.metadata.get("ocr_text", "")
        if ocr_text:
            # 如果有OCR文本，使用它作为内容并分块
            original_content = document.page_content
            document.page_content = ocr_text
            chunk_docs = self.document_processor.process_document(document)
            
            # 恢复原始内容到第一个分块
            if chunk_docs:
                chunk_docs[0].page_content = f"[IMAGE] {original_content}\n\nOCR文本: {ocr_text}"
            
            return chunk_docs
        else:
            # 没有OCR文本，返回原始文档
            return [document]
    
    def process_multimodal_documents(self, documents: List[Document]) -> List[Document]:
        """
        处理多个多模态文档
        """
        if not self.multimodal_enabled:
            # 如果多模态未启用，使用常规处理器
            return self.document_processor.process_documents(documents)
        
        processed_docs = []
        for doc in documents:
            chunk_docs = self.process_multimodal_document(doc)
            processed_docs.extend(chunk_docs)
        
        # 过滤短文档
        return self.document_processor.filter_documents(processed_docs)

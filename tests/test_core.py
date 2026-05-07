#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块测试
验证文档加载、处理、嵌入和向量存储组件。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document


class TestDocumentProcessor(unittest.TestCase):
    """测试 DocumentProcessor"""

    @patch("ragify.core.document_processors.get_config")
    def test_clean_text(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "retrieval.chunk_size": 100,
            "retrieval.chunk_overlap": 10,
        }.get(key, default)

        from ragify.core.document_processors import DocumentProcessor

        processor = DocumentProcessor()
        text = "Hello    world\n\n\nTest   text"
        cleaned = processor.clean_text(text)
        self.assertNotIn("    ", cleaned)
        self.assertIn("Hello world", cleaned)

    @patch("ragify.core.document_processors.get_config")
    def test_process_document_creates_chunks(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "retrieval.chunk_size": 20,
            "retrieval.chunk_overlap": 5,
        }.get(key, default)

        from ragify.core.document_processors import DocumentProcessor

        processor = DocumentProcessor()
        doc = Document(page_content="This is a test document with enough content to be split into multiple chunks.", metadata={"source": "test.txt"})
        chunks = processor.process_document(doc)
        self.assertGreater(len(chunks), 0)
        self.assertIn("chunk_index", chunks[0].metadata)
        self.assertIn("chunk_total", chunks[0].metadata)

    @patch("ragify.core.document_processors.get_config")
    def test_filter_documents(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "retrieval.chunk_size": 100,
            "retrieval.chunk_overlap": 10,
        }.get(key, default)

        from ragify.core.document_processors import DocumentProcessor

        processor = DocumentProcessor()
        docs = [
            Document(page_content="Short", metadata={}),
            Document(page_content="This is a longer document that should pass the filter.", metadata={}),
        ]
        filtered = processor.filter_documents(docs, min_length=10)
        self.assertEqual(len(filtered), 1)


class TestMultiModalDocumentProcessor(unittest.TestCase):
    """测试 MultiModalDocumentProcessor"""

    @patch("ragify.core.document_processors.get_config")
    def test_does_not_mutate_original_document(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "retrieval.chunk_size": 50,
            "retrieval.chunk_overlap": 5,
            "multimodal.enabled": True,
        }.get(key, default)

        from ragify.core.document_processors import MultiModalDocumentProcessor

        processor = MultiModalDocumentProcessor()
        original_content = "Original image content"
        doc = Document(
            page_content=original_content,
            metadata={"file_type": ".jpg", "ocr_text": "OCR extracted text"},
        )
        original_metadata = doc.metadata.copy()

        result = processor.process_multimodal_document(doc)

        # 验证原始对象未被修改
        self.assertEqual(doc.page_content, original_content)
        self.assertEqual(doc.metadata, original_metadata)
        # 验证结果包含多模态标记
        self.assertTrue(any("is_multimodal" in d.metadata for d in result))

    @patch("ragify.core.document_processors.get_config")
    def test_non_multimodal_fallback(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "retrieval.chunk_size": 50,
            "retrieval.chunk_overlap": 5,
            "multimodal.enabled": True,
        }.get(key, default)

        from ragify.core.document_processors import MultiModalDocumentProcessor

        processor = MultiModalDocumentProcessor()
        doc = Document(page_content="Plain text", metadata={"file_type": ".txt"})
        result = processor.process_multimodal_document(doc)
        self.assertGreater(len(result), 0)


class TestEmbeddingGenerator(unittest.TestCase):
    """测试 EmbeddingGenerator"""

    @patch("ragify.core.embeddings.get_config")
    def test_generate_embeddings_filters_empty(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "embeddings.provider": "openai",
            "embeddings.model_name": "text-embedding-3-small",
            "embeddings.dimensions": 1536,
        }.get(key, default)

        from ragify.core.embeddings import EmbeddingGenerator

        with patch.object(EmbeddingGenerator, "_initialize_embeddings") as mock_init:
            mock_embed = MagicMock()
            mock_embed.embed_documents.return_value = [[0.1, 0.2, 0.3]]
            mock_init.return_value = mock_embed

            gen = EmbeddingGenerator()
            texts = ["", "hello", "   "]
            result = gen.generate_embeddings(texts)
            # 空文本应被过滤，只有 "hello" 生成嵌入
            self.assertEqual(len(result), 3)
            zero_vec = [0.0] * 1536  # 与配置中的 dimensions 一致
            self.assertEqual(result[0], zero_vec)  # 空文本返回零向量
            self.assertEqual(result[1], [0.1, 0.2, 0.3])
            self.assertEqual(result[2], zero_vec)

    @patch("ragify.core.embeddings.get_config")
    def test_generate_single_embedding_empty(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "embeddings.provider": "openai",
            "embeddings.model_name": "text-embedding-3-small",
            "embeddings.dimensions": 1536,
        }.get(key, default)

        from ragify.core.embeddings import EmbeddingGenerator

        with patch.object(EmbeddingGenerator, "_initialize_embeddings") as mock_init:
            mock_init.return_value = MagicMock()
            gen = EmbeddingGenerator()
            result = gen.generate_single_embedding("   ")
            self.assertIsNone(result)


class TestVectorStoreManager(unittest.TestCase):
    """测试 VectorStoreManager"""

    @patch("ragify.core.vectorstores.get_config")
    @patch("ragify.core.vectorstores.EmbeddingGenerator")
    def test_add_documents_empty_list(self, mock_embed, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "vectorstore.type": "chromadb",
            "vectorstore.persist_directory": "./test_vectorstore",
            "vectorstore.collection_name": "test",
        }.get(key, default)

        from ragify.core.vectorstores import VectorStoreManager

        vsm = VectorStoreManager()
        result = vsm.add_documents([])
        self.assertEqual(result, [])

    @patch("ragify.core.vectorstores.get_config")
    @patch("ragify.core.vectorstores.EmbeddingGenerator")
    def test_similarity_search_no_vectorstore(self, mock_embed, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "vectorstore.type": "faiss",
            "vectorstore.persist_directory": "./test_vectorstore",
            "vectorstore.collection_name": "test",
        }.get(key, default)

        from ragify.core.vectorstores import VectorStoreManager

        vsm = VectorStoreManager()
        vsm.vectorstore = None
        result = vsm.similarity_search("test")
        self.assertEqual(result, [])


class TestDocumentLoader(unittest.TestCase):
    """测试文档加载器"""

    @patch("ragify.core.document_loaders.get_config")
    def test_load_file_not_found(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "multimodal.enabled": True,
            "multimodal.image_processor.enabled": True,
        }.get(key, default)

        from ragify.core.document_loaders import MultiModalDocumentLoader

        loader = MultiModalDocumentLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load_file("/nonexistent/file.txt")

    @patch("ragify.core.document_loaders.get_config")
    def test_load_directory_not_found(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "multimodal.enabled": True,
            "multimodal.image_processor.enabled": True,
        }.get(key, default)

        from ragify.core.document_loaders import MultiModalDocumentLoader

        loader = MultiModalDocumentLoader()
        with self.assertRaises(NotADirectoryError):
            loader.load_directory("/nonexistent/dir")

    @patch("ragify.core.document_loaders.get_config")
    def test_load_file_creates_new_document(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "multimodal.enabled": True,
            "multimodal.image_processor.enabled": True,
        }.get(key, default)

        from ragify.core.document_loaders import MultiModalDocumentLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world")
            path = f.name

        try:
            loader = MultiModalDocumentLoader()
            docs = loader.load_file(path)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "Hello world")
            self.assertEqual(docs[0].metadata["file_type"], ".txt")
            self.assertEqual(docs[0].metadata["source"], path)
        finally:
            os.unlink(path)


class TestLanguageModelManager(unittest.TestCase):
    """测试 LanguageModelManager"""

    @patch("ragify.core.language_models.get_config")
    def test_create_rag_prompt_contains_context(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "openai",
            "llm.model_name": "gpt-4",
            "llm.temperature": 0.7,
            "llm.max_tokens": 1024,
        }.get(key, default)

        from ragify.core.language_models import LanguageModelManager

        with patch.object(LanguageModelManager, "_initialize_llm") as mock_init:
            mock_init.return_value = MagicMock()
            lmm = LanguageModelManager()
            docs = [
                Document(page_content="Context 1", metadata={"source": "a.txt"}),
                Document(page_content="Context 2", metadata={"source": "b.txt"}),
            ]
            prompt = lmm.create_rag_prompt("What is this?", docs)
            self.assertIn("Context 1", prompt)
            self.assertIn("Context 2", prompt)
            self.assertIn("What is this?", prompt)

    @patch("ragify.core.language_models.get_config")
    def test_generate_response_no_llm(self, mock_config):
        mock_config.return_value = MagicMock()
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            "llm.provider": "openai",
            "llm.model_name": "gpt-4",
            "llm.temperature": 0.7,
            "llm.max_tokens": 1024,
        }.get(key, default)

        from ragify.core.language_models import LanguageModelManager

        with patch.object(LanguageModelManager, "_initialize_llm") as mock_init:
            mock_init.return_value = None
            lmm = LanguageModelManager()
            result = lmm.generate_response("test")
            self.assertIn("未初始化", result)


if __name__ == "__main__":
    unittest.main()

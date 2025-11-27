from typing import List, Dict, Any, Optional, Union
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from ..config import get_config
import numpy as np


class EmbeddingGenerator:
    """
    嵌入生成器，负责生成文本嵌入向量
    """
    
    def __init__(self):
        self.config = get_config()
        self.embedding_provider = self.config.get("embeddings.provider", "openai")
        self.embedding_model = self.config.get("embeddings.model_name", "text-embedding-3-small")
        self.embedding_dimensions = self.config.get("embeddings.dimensions", 1536)
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> Embeddings:
        """
        初始化嵌入模型
        """
        if self.embedding_provider == "openai":
            # 初始化OpenAI嵌入
            api_key = self.config.get("embeddings.api_key")
            return OpenAIEmbeddings(
                model=self.embedding_model,
                dimensions=self.embedding_dimensions,
                api_key=api_key
            )
        elif self.embedding_provider == "sentence_transformers":
            # 可以添加Sentence Transformers支持
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=self.embedding_model)
            except ImportError:
                raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        else:
            # 默认使用OpenAI
            print(f"警告: 不支持的嵌入提供者 {self.embedding_provider}，使用OpenAI默认值")
            return OpenAIEmbeddings(
                model=self.embedding_model,
                dimensions=self.embedding_dimensions
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        为文本列表生成嵌入向量
        """
        # 过滤空文本
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            return []
        
        try:
            embeddings = self.embeddings.embed_documents(valid_texts)
            
            # 映射回原始顺序
            result = []
            valid_idx = 0
            for text in texts:
                if text.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # 为空文本生成零向量
                    result.append([0.0] * self.embedding_dimensions)
            
            return result
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}")
            return []
    
    def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        为单个文本生成嵌入向量
        """
        if not text.strip():
            return None
        
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"生成单个嵌入向量时出错: {e}")
            return None


class MultiModalEmbeddingGenerator(EmbeddingGenerator):
    """
    多模态嵌入生成器，支持文本和图像等多模态内容的嵌入
    """
    
    def __init__(self):
        super().__init__()
        self.multimodal_enabled = self.config.get("multimodal.enabled", True)
    
    def is_multimodal_content(self, content: Union[str, Dict[str, Any]]) -> bool:
        """
        判断内容是否为多模态
        """
        if isinstance(content, dict):
            return any(key in content for key in ["image", "image_path", "audio", "video"])
        return False
    
    def generate_multimodal_embedding(self, content: Union[str, Dict[str, Any]]) -> Optional[List[float]]:
        """
        生成多模态内容的嵌入向量
        """
        if not self.multimodal_enabled:
            # 如果多模态未启用，回退到文本嵌入
            if isinstance(content, dict):
                text = content.get("text", "") or str(content)
                return self.generate_single_embedding(text)
            return self.generate_single_embedding(str(content))
        
        if isinstance(content, str):
            # 纯文本，使用常规嵌入
            return self.generate_single_embedding(content)
        elif isinstance(content, dict):
            # 多模态内容
            # 这里我们简单地将文本描述作为嵌入
            # 在实际应用中，可能需要使用专门的多模态嵌入模型
            text_representation = self._get_text_representation(content)
            return self.generate_single_embedding(text_representation)
        
        return None
    
    def _get_text_representation(self, multimodal_content: Dict[str, Any]) -> str:
        """
        为多模态内容生成文本表示
        """
        parts = []
        
        # 添加文本部分
        if "text" in multimodal_content:
            parts.append(multimodal_content["text"])
        
        # 添加图像描述
        if "image_path" in multimodal_content:
            parts.append(f"[图像文件: {multimodal_content['image_path']}]")
        elif "image" in multimodal_content:
            parts.append("[图像内容]")
        
        # 添加其他模态信息
        if "audio" in multimodal_content:
            parts.append("[音频内容]")
        if "video" in multimodal_content:
            parts.append("[视频内容]")
        
        # 如果没有任何描述，使用默认文本
        if not parts:
            return str(multimodal_content)
        
        return " ".join(parts)
    
    def combine_embeddings(self, embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """
        组合多个嵌入向量
        用于多模态内容的嵌入融合
        """
        if not embeddings:
            return []
        
        # 转换为numpy数组
        embeddings_array = np.array(embeddings)
        
        if weights is None:
            # 等权重平均
            combined = np.mean(embeddings_array, axis=0)
        else:
            # 加权平均
            if len(weights) != len(embeddings):
                raise ValueError("权重数量必须与嵌入数量相同")
            # 归一化权重
            normalized_weights = np.array(weights) / np.sum(weights)
            combined = np.average(embeddings_array, axis=0, weights=normalized_weights)
        
        return combined.tolist()

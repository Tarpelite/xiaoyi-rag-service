"""
Embedding 服务
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


class EmbeddingService:
    """Embedding 生成服务"""

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model: Optional[SentenceTransformer] = None
        self.device = settings.embedding_device
        self._initialized = True

    def _download_from_modelscope(self, model_id: str) -> str:
        """
        从 ModelScope 下载模型
        
        Args:
            model_id: ModelScope 模型 ID，如 "AI-ModelScope/bge-large-zh-v1.5"
            
        Returns:
            本地模型路径
        """
        from modelscope import snapshot_download
        
        logger.info(f"从 ModelScope 下载模型: {model_id}")
        
        # 下载到缓存目录
        cache_dir = settings.hf_cache_dir or Path.home() / ".cache" / "modelscope"
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
        )
        
        logger.info(f"模型下载完成: {model_dir}")
        return model_dir

    def load_model(self):
        """加载 Embedding 模型"""
        if self.model is not None:
            return

        # 根据配置选择模型来源
        if settings.model_source == "modelscope":
            model_path = self._download_from_modelscope(settings.embedding_model)
        else:
            model_path = settings.embedding_model_hf
            
        logger.info(f"加载 Embedding 模型: {model_path} on {self.device}")

        self.model = SentenceTransformer(
            model_path,
            device=self.device,
        )

        # 预热
        self.model.encode(["warmup"], convert_to_tensor=True)

        logger.info(f"Embedding 模型加载完成, 维度: {self.model.get_sentence_embedding_dimension()}")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def encode(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        批量生成 Embedding

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条

        Returns:
            向量列表
        """
        if not self.is_loaded():
            self.load_model()

        batch_size = batch_size or settings.embedding_batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 归一化，用于余弦相似度
        )

        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """
        对单个查询生成 Embedding

        Args:
            query: 查询文本

        Returns:
            查询向量
        """
        # BGE 模型的查询需要加前缀
        if "bge" in settings.embedding_model.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"

        return self.encode([query])[0]


def get_embedding_service() -> EmbeddingService:
    """获取 Embedding 服务单例"""
    return EmbeddingService()

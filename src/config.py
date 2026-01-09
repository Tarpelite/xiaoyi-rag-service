"""
配置管理模块
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ========== Qdrant 配置 ==========
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "reports"
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True

    # ========== Embedding 配置 ==========
    # ModelScope 模型 ID（国内推荐）
    embedding_model: str = "AI-ModelScope/bge-large-zh-v1.5"
    # HuggingFace 模型 ID（备用）
    embedding_model_hf: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cuda"  # cuda / cpu
    embedding_batch_size: int = 32
    embedding_dim: int = 1024  # bge-large-zh 的维度

    # ========== Reranker 配置 ==========
    # ModelScope 模型 ID（国内推荐）
    reranker_model: str = "BAAI/bge-reranker-large"
    # HuggingFace 模型 ID（备用）
    reranker_model_hf: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cuda"
    reranker_batch_size: int = 32

    # ========== 模型来源 ==========
    # 可选: "modelscope" (国内推荐) 或 "huggingface"
    model_source: str = "modelscope"

    # ========== 检索配置 ==========
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    top_k_retrieval: int = 100  # 初筛数量
    top_k_rerank: int = 10  # Rerank 后返回数量

    # ========== 分块配置 ==========
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ========== 路径配置 ==========
    data_dir: Path = Path("./data")
    pdf_dir: Path = Path("./data/pdfs")
    parsed_dir: Path = Path("./data/parsed")

    # ========== API 配置 ==========
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # ========== 日志配置 ==========
    log_level: str = "INFO"

    # ========== 模型缓存 ==========
    hf_cache_dir: Optional[str] = None  # HuggingFace 模型缓存目录

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def qdrant_url(self) -> str:
        """Qdrant HTTP URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 导出便捷访问
settings = get_settings()
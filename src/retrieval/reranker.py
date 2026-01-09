"""
Reranker 重排序服务
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


class RerankerService:
    """Reranker 重排序服务"""

    _instance: Optional["RerankerService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.device = settings.reranker_device
        self._initialized = True

    def _download_from_modelscope(self, model_id: str) -> str:
        """
        从 ModelScope 下载模型
        
        Args:
            model_id: ModelScope 模型 ID
            
        Returns:
            本地模型路径
        """
        from modelscope import snapshot_download
        
        logger.info(f"从 ModelScope 下载 Reranker 模型: {model_id}")
        
        cache_dir = settings.hf_cache_dir or Path.home() / ".cache" / "modelscope"
        model_dir = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
        )
        
        logger.info(f"Reranker 模型下载完成: {model_dir}")
        return model_dir

    def load_model(self):
        """加载 Reranker 模型"""
        if self.model is not None:
            return

        # 根据配置选择模型来源
        if settings.model_source == "modelscope":
            model_path = self._download_from_modelscope(settings.reranker_model)
        else:
            model_path = settings.reranker_model_hf

        logger.info(f"加载 Reranker 模型: {model_path} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")
            if self.device == "cuda":
                logger.warning("CUDA 不可用，使用 CPU")
                self.device = "cpu"

        self.model.eval()

        logger.info("Reranker 模型加载完成")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10,
        score_key: str = "content",
    ) -> list[dict]:
        """
        对检索结果重排序

        Args:
            query: 查询文本
            documents: 文档列表，需要包含 score_key 指定的字段
            top_k: 返回数量
            score_key: 用于排序的文本字段

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        if not self.is_loaded():
            self.load_model()

        # 构建 query-document 对
        pairs = [[query, doc[score_key]] for doc in documents]

        # 批量计算得分
        scores = self._compute_scores(pairs)

        # 添加 rerank_score 并排序
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score

        # 按 rerank_score 排序
        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

        return sorted_docs[:top_k]

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        """批量计算 rerank 分数"""
        scores = []

        # 分批处理
        batch_size = settings.reranker_batch_size
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            batch_scores = self._score_batch(batch)
            scores.extend(batch_scores)

        return scores

    def _score_batch(self, pairs: list[list[str]]) -> list[float]:
        """对一批 query-document 对计算分数"""
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1)

            # Sigmoid 转换为 0-1 分数
            scores = torch.sigmoid(scores)

            return scores.cpu().tolist()


def get_reranker_service() -> RerankerService:
    """获取 Reranker 服务单例"""
    return RerankerService()

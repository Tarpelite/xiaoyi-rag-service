"""
BM25 关键词检索服务
"""

from typing import Optional

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

from src.storage import get_qdrant_storage


class BM25Retriever:
    """BM25 关键词检索"""

    _instance: Optional["BM25Retriever"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.bm25: Optional[BM25Okapi] = None
        self.corpus: list[dict] = []  # 原始文档数据
        self.tokenized_corpus: list[list[str]] = []  # 分词后的语料
        self._initialized = True

    def build_index(self, force_rebuild: bool = False):
        """
        构建 BM25 索引

        Args:
            force_rebuild: 是否强制重建
        """
        if self.bm25 is not None and not force_rebuild:
            logger.info("BM25 索引已存在，跳过构建")
            return

        logger.info("开始构建 BM25 索引...")

        # 从 Qdrant 获取所有分块
        storage = get_qdrant_storage()
        self.corpus = storage.get_all_chunks_for_bm25()

        if not self.corpus:
            logger.warning("没有找到任何分块，BM25 索引为空")
            self.bm25 = None
            return

        # 分词
        self.tokenized_corpus = [self._tokenize(doc["content"]) for doc in self.corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"BM25 索引构建完成，共 {len(self.corpus)} 个分块")

    def _tokenize(self, text: str) -> list[str]:
        """中文分词"""
        # 使用 jieba 分词，过滤停用词和短词
        tokens = jieba.lcut(text)
        # 过滤：只保留长度 > 1 的词
        return [t for t in tokens if len(t) > 1]

    def search(self, query: str, top_k: int = 100) -> list[dict]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        if self.bm25 is None:
            logger.warning("BM25 索引未构建，尝试构建...")
            self.build_index()

        if self.bm25 is None:
            return []

        # 对查询分词
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # 获取 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 获取 top_k 结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有得分的结果
                doc = self.corpus[idx]
                results.append(
                    {
                        "chunk_id": doc["chunk_id"],
                        "doc_id": doc["doc_id"],
                        "content": doc["content"],
                        "page_number": doc["page_number"],
                        "file_name": doc["file_name"],
                        "title": doc.get("title"),
                        "score": float(scores[idx]),
                    }
                )

        return results

    def is_ready(self) -> bool:
        """检查 BM25 索引是否就绪"""
        return self.bm25 is not None and len(self.corpus) > 0


def get_bm25_retriever() -> BM25Retriever:
    """获取 BM25 检索器单例"""
    return BM25Retriever()

"""
混合检索服务
===============

支持三种检索模式：
1. HYBRID: BM25 + 向量检索 + Rerank（默认）
2. VECTOR: 仅向量检索
3. BM25: 仅关键词检索
"""

import time
from typing import Optional

from loguru import logger

from src.config import settings
from src.models import SearchFilters, SearchMode, SearchRequest, SearchResponse, SearchResultItem
from src.storage import get_qdrant_storage

from .bm25 import get_bm25_retriever
from .embedding import get_embedding_service
from .reranker import get_reranker_service


class HybridRetriever:
    """混合检索器"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.bm25_retriever = get_bm25_retriever()
        self.reranker_service = get_reranker_service()
        self.qdrant_storage = get_qdrant_storage()

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        执行检索

        Args:
            request: 搜索请求

        Returns:
            搜索响应
        """
        start_time = time.time()

        # 根据模式选择检索策略
        if request.mode == SearchMode.VECTOR:
            results = self._vector_search(
                request.query, request.top_k * 2, request.filters  # 多取一些用于去重
            )
        elif request.mode == SearchMode.BM25:
            results = self._bm25_search(request.query, request.top_k * 2)
        else:  # HYBRID
            results = self._hybrid_search(
                request.query,
                request.bm25_weight or settings.bm25_weight,
                request.vector_weight or settings.vector_weight,
                settings.top_k_retrieval,
                request.filters,
            )

        # Rerank（如果启用）
        used_rerank = False
        if request.use_rerank and results:
            results = self.reranker_service.rerank(
                request.query, results, top_k=request.top_k
            )
            used_rerank = True
        else:
            results = results[: request.top_k]

        # 转换为响应格式
        result_items = [
            SearchResultItem(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                content=r["content"],
                score=r.get("rerank_score", r.get("score", 0)),
                page_number=r["page_number"],
                file_name=r["file_name"],
                title=r.get("title"),
                section_title=r.get("section_title"),
                bm25_score=r.get("bm25_score"),
                vector_score=r.get("vector_score"),
                rerank_score=r.get("rerank_score"),
            )
            for r in results
        ]

        took_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            total=len(result_items),
            results=result_items,
            mode=request.mode,
            took_ms=took_ms,
            used_rerank=used_rerank,
        )

    def _vector_search(
        self, query: str, top_k: int, filters: Optional[SearchFilters] = None
    ) -> list[dict]:
        """向量检索"""
        query_vector = self.embedding_service.encode_query(query)

        filters_dict = self._convert_filters(filters)
        results = self.qdrant_storage.search_vector(query_vector, top_k, filters_dict)

        # 标记分数来源
        for r in results:
            r["vector_score"] = r["score"]

        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 关键词检索"""
        results = self.bm25_retriever.search(query, top_k)

        # 标记分数来源
        for r in results:
            r["bm25_score"] = r["score"]

        return results

    def _hybrid_search(
        self,
        query: str,
        bm25_weight: float,
        vector_weight: float,
        top_k: int,
        filters: Optional[SearchFilters] = None,
    ) -> list[dict]:
        """
        混合检索

        使用 RRF (Reciprocal Rank Fusion) 融合 BM25 和向量检索结果
        """
        # 分别检索
        vector_results = self._vector_search(query, top_k, filters)
        bm25_results = self._bm25_search(query, top_k)

        # RRF 融合
        k = 60  # RRF 参数
        scores = {}  # chunk_id -> {score, data}

        # 处理向量检索结果
        for rank, r in enumerate(vector_results):
            chunk_id = r["chunk_id"]
            rrf_score = vector_weight / (k + rank + 1)

            if chunk_id not in scores:
                scores[chunk_id] = {"score": 0, "data": r}

            scores[chunk_id]["score"] += rrf_score
            scores[chunk_id]["data"]["vector_score"] = r.get("vector_score", r["score"])

        # 处理 BM25 结果
        for rank, r in enumerate(bm25_results):
            chunk_id = r["chunk_id"]
            rrf_score = bm25_weight / (k + rank + 1)

            if chunk_id not in scores:
                scores[chunk_id] = {"score": 0, "data": r}

            scores[chunk_id]["score"] += rrf_score
            scores[chunk_id]["data"]["bm25_score"] = r.get("bm25_score", r["score"])

        # 排序并返回
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for item in sorted_items[:top_k]:
            data = item["data"]
            data["score"] = item["score"]  # 使用 RRF 融合分数
            results.append(data)

        return results

    def _convert_filters(self, filters: Optional[SearchFilters]) -> Optional[dict]:
        """转换过滤条件格式"""
        if not filters:
            return None

        return {
            "doc_ids": filters.doc_ids,
            "industries": filters.industries,
            "stock_codes": filters.stock_codes,
            "report_types": filters.report_types,
        }

    def init_services(self, load_models: bool = True):
        """
        初始化所有服务

        Args:
            load_models: 是否预加载模型
        """
        logger.info("初始化检索服务...")

        # 检查 Qdrant 连接
        if not self.qdrant_storage.is_connected():
            raise RuntimeError("无法连接到 Qdrant")

        # 构建 BM25 索引
        self.bm25_retriever.build_index()

        if load_models:
            # 预加载 Embedding 模型
            self.embedding_service.load_model()

            # 预加载 Reranker 模型
            self.reranker_service.load_model()

        logger.info("检索服务初始化完成")


def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器"""
    return HybridRetriever()

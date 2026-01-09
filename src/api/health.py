"""
健康检查 API 路由
"""

from fastapi import APIRouter

from src.models import HealthResponse, StatsResponse
from src.retrieval import get_bm25_retriever, get_embedding_service, get_reranker_service
from src.storage import get_qdrant_storage

router = APIRouter(tags=["Health"])

VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    健康检查

    返回服务状态、模型加载状态、数据库连接状态等信息。
    """
    qdrant = get_qdrant_storage()
    embedding = get_embedding_service()
    reranker = get_reranker_service()

    # 获取 collection 信息
    collection_info = qdrant.get_collection_info()

    return HealthResponse(
        status="healthy",
        version=VERSION,
        qdrant_connected=qdrant.is_connected(),
        embedding_model_loaded=embedding.is_loaded(),
        reranker_model_loaded=reranker.is_loaded(),
        total_documents=0,  # TODO: 从元数据存储获取
        total_chunks=collection_info.get("points_count", 0),
        index_size_mb=0,  # TODO: 计算实际大小
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    获取统计信息

    返回文档数量、分块数量、行业分布等信息。
    """
    qdrant = get_qdrant_storage()
    collection_info = qdrant.get_collection_info()

    return StatsResponse(
        total_documents=0,  # TODO: 从元数据存储获取
        total_chunks=collection_info.get("points_count", 0),
        index_size_mb=0,
        industries=[],  # TODO: 聚合查询
        report_types=[],
    )


@router.get("/ready")
async def readiness_check():
    """
    就绪检查

    用于 Kubernetes 等编排系统判断服务是否可以接受流量。
    """
    qdrant = get_qdrant_storage()

    if not qdrant.is_connected():
        return {"ready": False, "reason": "Qdrant not connected"}

    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    存活检查

    用于 Kubernetes 等编排系统判断服务是否存活。
    """
    return {"alive": True}

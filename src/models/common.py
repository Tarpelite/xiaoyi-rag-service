"""
通用数据模型
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    qdrant_connected: bool = Field(..., description="Qdrant 连接状态")
    embedding_model_loaded: bool = Field(..., description="Embedding 模型是否加载")
    reranker_model_loaded: bool = Field(..., description="Reranker 模型是否加载")

    # 统计信息
    total_documents: int = Field(0, description="文档总数")
    total_chunks: int = Field(0, description="分块总数")
    index_size_mb: float = Field(0, description="索引大小(MB)")


class StatsResponse(BaseModel):
    """统计信息响应"""

    total_documents: int
    total_chunks: int
    index_size_mb: float
    industries: list[str] = Field(default_factory=list, description="所有行业分类")
    report_types: list[str] = Field(default_factory=list, description="所有研报类型")


class ErrorResponse(BaseModel):
    """错误响应"""

    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    detail: Optional[Any] = Field(None, description="详细信息")


class SuccessResponse(BaseModel):
    """通用成功响应"""

    success: bool = True
    message: str
    data: Optional[Any] = None

"""
API 路由模块
"""

from fastapi import APIRouter

from .documents import router as documents_router
from .health import router as health_router
from .search import router as search_router

# 创建主路由
api_router = APIRouter(prefix="/api/v1")

# 注册子路由
api_router.include_router(search_router)
api_router.include_router(documents_router)
api_router.include_router(health_router)

__all__ = ["api_router"]

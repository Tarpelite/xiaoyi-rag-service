"""
xiaoyi-rag-service 主入口
=========================

研报知识库检索服务
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api import api_router
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("=" * 50)
    logger.info("xiaoyi-rag-service 启动中...")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"Embedding: {settings.embedding_model} on {settings.embedding_device}")
    logger.info(f"Reranker: {settings.reranker_model} on {settings.reranker_device}")
    logger.info("=" * 50)

    # 初始化服务（可选预加载模型）
    if not settings.debug:
        try:
            from src.retrieval import get_hybrid_retriever

            retriever = get_hybrid_retriever()
            retriever.init_services(load_models=True)
            logger.info("✅ 服务初始化完成")
        except Exception as e:
            logger.warning(f"⚠️ 服务初始化失败（将在首次请求时重试）: {e}")

    yield

    # 关闭时
    logger.info("xiaoyi-rag-service 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="xiaoyi-rag-service",
    description="研报知识库检索服务 - 为 xiaoyi 提供高质量的研报检索能力",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router)


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "xiaoyi-rag-service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )

"""
搜索 API 路由
"""

from fastapi import APIRouter, HTTPException

from src.models import SearchRequest, SearchResponse
from src.retrieval import get_hybrid_retriever

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    混合检索接口

    支持三种检索模式：
    - **hybrid**: BM25 + 向量检索 + Rerank（默认，效果最好）
    - **vector**: 仅向量语义检索
    - **bm25**: 仅关键词检索

    **示例请求:**
    ```json
    {
        "query": "2024年新能源行业发展趋势",
        "top_k": 5,
        "mode": "hybrid",
        "use_rerank": true
    }
    ```

    **带过滤条件:**
    ```json
    {
        "query": "锂电池产能",
        "top_k": 10,
        "filters": {
            "industries": ["新能源", "锂电池"],
            "report_types": ["行业研究"]
        }
    }
    ```
    """
    try:
        retriever = get_hybrid_retriever()
        return retriever.search(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

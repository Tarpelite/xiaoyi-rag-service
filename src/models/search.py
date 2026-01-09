"""
搜索相关的数据模型
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    """搜索模式"""

    HYBRID = "hybrid"  # 混合检索（默认）
    VECTOR = "vector"  # 仅向量检索
    BM25 = "bm25"  # 仅关键词检索


class SearchRequest(BaseModel):
    """搜索请求"""

    query: str = Field(..., description="搜索查询", min_length=1, max_length=1000)
    top_k: int = Field(5, description="返回结果数量", ge=1, le=100)
    mode: SearchMode = Field(SearchMode.HYBRID, description="搜索模式")

    # 可选过滤条件
    filters: Optional["SearchFilters"] = Field(None, description="过滤条件")

    # 高级参数
    use_rerank: bool = Field(True, description="是否使用 Reranker")
    bm25_weight: Optional[float] = Field(None, description="BM25 权重 (0-1)")
    vector_weight: Optional[float] = Field(None, description="向量权重 (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "2024年新能源行业发展趋势",
                "top_k": 5,
                "mode": "hybrid",
                "use_rerank": True,
            }
        }


class SearchFilters(BaseModel):
    """搜索过滤条件"""

    doc_ids: Optional[list[str]] = Field(None, description="限定文档 ID 列表")
    industries: Optional[list[str]] = Field(None, description="限定行业")
    stock_codes: Optional[list[str]] = Field(None, description="限定股票代码")
    report_types: Optional[list[str]] = Field(None, description="限定研报类型")
    date_from: Optional[str] = Field(None, description="发布日期起始 (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="发布日期截止 (YYYY-MM-DD)")


class SearchResultItem(BaseModel):
    """单条搜索结果"""

    chunk_id: str = Field(..., description="分块 ID")
    doc_id: str = Field(..., description="文档 ID")
    content: str = Field(..., description="分块内容")
    score: float = Field(..., description="相关度得分")
    page_number: int = Field(..., description="所在页码")

    # 文档元数据
    file_name: str = Field(..., description="文件名")
    title: Optional[str] = Field(None, description="文档标题")
    section_title: Optional[str] = Field(None, description="章节标题")

    # 得分详情（调试用）
    bm25_score: Optional[float] = Field(None, description="BM25 得分")
    vector_score: Optional[float] = Field(None, description="向量得分")
    rerank_score: Optional[float] = Field(None, description="Rerank 得分")


class SearchResponse(BaseModel):
    """搜索响应"""

    query: str = Field(..., description="原始查询")
    total: int = Field(..., description="匹配结果总数")
    results: list[SearchResultItem] = Field(..., description="搜索结果列表")

    # 元信息
    mode: SearchMode = Field(..., description="使用的搜索模式")
    took_ms: float = Field(..., description="检索耗时(毫秒)")
    used_rerank: bool = Field(..., description="是否使用了 Rerank")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "新能源发展",
                "total": 42,
                "results": [
                    {
                        "chunk_id": "chunk_001",
                        "doc_id": "doc_001",
                        "content": "2024年新能源行业预计将保持高速增长...",
                        "score": 0.89,
                        "page_number": 5,
                        "file_name": "新能源行业2024展望.pdf",
                        "title": "新能源行业2024年度展望报告",
                    }
                ],
                "mode": "hybrid",
                "took_ms": 123.45,
                "used_rerank": True,
            }
        }

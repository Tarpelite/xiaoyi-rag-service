"""
文档相关的数据模型
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """文档元数据"""

    doc_id: str = Field(..., description="文档唯一 ID")
    file_name: str = Field(..., description="原始文件名")
    file_path: Optional[str] = Field(None, description="文件路径")
    title: Optional[str] = Field(None, description="文档标题")
    author: Optional[str] = Field(None, description="作者/机构")
    publish_date: Optional[str] = Field(None, description="发布日期")
    source: Optional[str] = Field(None, description="来源")
    total_pages: int = Field(0, description="总页数")
    total_chunks: int = Field(0, description="分块总数")
    file_size: Optional[int] = Field(None, description="文件大小(字节)")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")

    # 研报特有字段
    industry: Optional[str] = Field(None, description="行业分类")
    stock_codes: list[str] = Field(default_factory=list, description="相关股票代码")
    report_type: Optional[str] = Field(None, description="研报类型（行业研究/个股研究/宏观研究等）")


class Chunk(BaseModel):
    """文档分块"""

    chunk_id: str = Field(..., description="分块唯一 ID")
    doc_id: str = Field(..., description="所属文档 ID")
    content: str = Field(..., description="分块内容")
    page_number: int = Field(..., description="所在页码")
    chunk_index: int = Field(..., description="在文档中的序号")

    # 可选元数据
    section_title: Optional[str] = Field(None, description="所属章节标题")
    char_start: Optional[int] = Field(None, description="在原文中的起始位置")
    char_end: Optional[int] = Field(None, description="在原文中的结束位置")


class ChunkWithEmbedding(Chunk):
    """带向量的分块"""

    embedding: list[float] = Field(..., description="向量表示")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    doc_id: str
    file_name: str
    total_pages: int
    total_chunks: int
    message: str = "Document indexed successfully"


class DocumentInfo(BaseModel):
    """文档信息（用于列表展示）"""

    doc_id: str
    file_name: str
    title: Optional[str] = None
    total_pages: int
    total_chunks: int
    created_at: datetime
    industry: Optional[str] = None
    report_type: Optional[str] = None


class DocumentDetail(BaseModel):
    """文档详情"""

    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)

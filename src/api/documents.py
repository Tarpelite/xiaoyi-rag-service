"""
文档管理 API 路由
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from src.config import settings
from src.models import DocumentInfo, DocumentMetadata, DocumentUploadResponse, SuccessResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

# 简单的内存存储（生产环境应该用 SQLite）
_document_store: dict[str, DocumentMetadata] = {}


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF 文件"),
    title: Optional[str] = Form(None, description="文档标题"),
    industry: Optional[str] = Form(None, description="行业分类"),
    report_type: Optional[str] = Form(None, description="研报类型"),
) -> DocumentUploadResponse:
    """
    上传并索引文档

    接受 PDF 文件，会自动解析、分块、生成向量并建立索引。

    **注意**: 大文件处理可能需要较长时间。
    """
    # 检查文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 生成文档 ID
    doc_id = str(uuid.uuid4())

    # 保存文件
    file_path = settings.pdf_dir / f"{doc_id}_{file.filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"文件已保存: {file_path}")

        # TODO: 调用索引 pipeline
        # 目前只是占位，实际实现需要：
        # 1. 调用 MinerU 解析 PDF
        # 2. 文本分块
        # 3. 生成 Embedding
        # 4. 存入 Qdrant

        # 创建元数据（占位）
        metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=file.filename,
            file_path=str(file_path),
            title=title or file.filename.replace(".pdf", ""),
            industry=industry,
            report_type=report_type,
            total_pages=0,  # 实际解析后填充
            total_chunks=0,  # 实际索引后填充
        )

        _document_store[doc_id] = metadata

        return DocumentUploadResponse(
            doc_id=doc_id,
            file_name=file.filename,
            total_pages=metadata.total_pages,
            total_chunks=metadata.total_chunks,
            message="文档已上传，索引功能待实现",
        )

    except Exception as e:
        # 清理文件
        if file_path.exists():
            os.remove(file_path)
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.get("", response_model=list[DocumentInfo])
async def list_documents(
    industry: Optional[str] = None,
    report_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[DocumentInfo]:
    """
    获取文档列表

    支持按行业、研报类型过滤。
    """
    docs = list(_document_store.values())

    # 过滤
    if industry:
        docs = [d for d in docs if d.industry == industry]
    if report_type:
        docs = [d for d in docs if d.report_type == report_type]

    # 分页
    docs = docs[offset : offset + limit]

    return [
        DocumentInfo(
            doc_id=d.doc_id,
            file_name=d.file_name,
            title=d.title,
            total_pages=d.total_pages,
            total_chunks=d.total_chunks,
            created_at=d.created_at,
            industry=d.industry,
            report_type=d.report_type,
        )
        for d in docs
    ]


@router.get("/{doc_id}", response_model=DocumentMetadata)
async def get_document(doc_id: str) -> DocumentMetadata:
    """
    获取文档详情
    """
    if doc_id not in _document_store:
        raise HTTPException(status_code=404, detail="文档不存在")

    return _document_store[doc_id]


@router.delete("/{doc_id}", response_model=SuccessResponse)
async def delete_document(doc_id: str) -> SuccessResponse:
    """
    删除文档

    会同时删除：
    - 原始 PDF 文件
    - 解析结果
    - 向量索引
    """
    if doc_id not in _document_store:
        raise HTTPException(status_code=404, detail="文档不存在")

    metadata = _document_store[doc_id]

    try:
        # 删除 PDF 文件
        if metadata.file_path and Path(metadata.file_path).exists():
            os.remove(metadata.file_path)

        # TODO: 删除 Qdrant 中的向量
        # storage = get_qdrant_storage()
        # storage.delete_document(doc_id)

        # 删除元数据
        del _document_store[doc_id]

        return SuccessResponse(message=f"文档 {doc_id} 已删除")

    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

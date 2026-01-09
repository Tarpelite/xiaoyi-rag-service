"""
文本分块器
"""

import re
import uuid
from typing import Optional

from loguru import logger

from src.config import settings
from src.models import Chunk


class TextChunker:
    """智能文本分块器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # 安全检查：overlap 不能大于 chunk_size 的一半
        if self.chunk_overlap >= self.chunk_size // 2:
            logger.warning(f"chunk_overlap ({self.chunk_overlap}) 过大，调整为 {self.chunk_size // 4}")
            self.chunk_overlap = self.chunk_size // 4

    def chunk_document(
        self,
        doc_id: str,
        pages: list[dict],
        doc_info: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        对文档进行分块

        Args:
            doc_id: 文档 ID
            pages: 页面列表，每个元素包含 page_number 和 text
            doc_info: 文档信息（可选）

        Returns:
            分块列表
        """
        chunks = []
        chunk_index = 0
        
        total_chars = sum(len(p.get("text", "")) for p in pages)
        logger.info(f"开始分块: {len(pages)} 页, 总字符数 {total_chars}")

        for page in pages:
            page_number = page["page_number"]
            text = page.get("text", "")
            
            if not text or not text.strip():
                continue

            # 清理文本
            text = self._clean_text(text)
            
            # 按段落分割
            paragraphs = self._split_paragraphs(text)

            # 合并小段落，分割大段落
            page_chunks = self._create_chunks(paragraphs)

            for chunk_text in page_chunks:
                if not chunk_text or not chunk_text.strip():
                    continue
                    
                chunk_id = f"{doc_id}_{chunk_index:04d}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        content=chunk_text.strip(),
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1
                
                # 安全检查：防止生成过多分块
                if chunk_index > 10000:
                    logger.warning(f"分块数量超过 10000，停止分块")
                    return chunks

        logger.info(f"分块完成: 共 {len(chunks)} 个分块")
        return chunks

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除过多的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> list[str]:
        """按段落分割文本"""
        # 按换行符分割
        paragraphs = re.split(r"\n{2,}", text)

        # 过滤空段落，并限制单个段落长度
        result = []
        for p in paragraphs:
            p = p.strip()
            if p:
                result.append(p)
        
        return result

    def _create_chunks(self, paragraphs: list[str]) -> list[str]:
        """
        创建分块

        - 小段落合并
        - 大段落分割
        - 保证 overlap
        """
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前段落太长，需要分割
            if len(para) > self.chunk_size:
                # 先保存当前累积的内容
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # 分割长段落
                sub_chunks = self._split_long_text(para)
                chunks.extend(sub_chunks)

            # 如果加上当前段落不超过限制，合并
            elif len(current_chunk) + len(para) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para

            # 否则，保存当前块，开始新块
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # 保留 overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + para if overlap_text else para

        # 保存最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        """分割长文本"""
        chunks = []
        start = 0
        max_iterations = len(text) // (self.chunk_size - self.chunk_overlap) + 10
        iterations = 0

        while start < len(text):
            iterations += 1
            if iterations > max_iterations:
                logger.warning(f"分割长文本迭代次数过多，强制退出")
                # 直接把剩余文本作为最后一个 chunk
                remaining = text[start:].strip()
                if remaining:
                    chunks.append(remaining)
                break
                
            end = start + self.chunk_size

            # 如果不是最后一块，尝试在句子边界分割
            if end < len(text):
                # 寻找句子结束符
                best_end = end
                for sep in ["。", "！", "？", ".", "!", "?", "；", ";", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + self.chunk_size // 2:  # 至少保留一半
                        best_end = last_sep + 1
                        break
                end = best_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 下一块的起始位置（考虑 overlap）
            if end >= len(text):
                break
            
            # 确保 start 一定前进
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = start + max(1, self.chunk_size // 2)
            start = new_start

        return chunks

    def _get_overlap(self, text: str) -> str:
        """获取 overlap 部分"""
        if not text or self.chunk_overlap <= 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        return text[-self.chunk_overlap :]


def get_chunker() -> TextChunker:
    """获取分块器"""
    return TextChunker()

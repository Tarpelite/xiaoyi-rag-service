"""
索引构建 Pipeline
==================

完整的文档索引流程：PDF 解析 → 分块 → Embedding → 存储
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from src.config import settings
from src.models import ChunkWithEmbedding, DocumentMetadata
from src.retrieval import get_embedding_service
from src.storage import get_qdrant_storage

from .chunker import get_chunker


class IndexingPipeline:
    """索引构建 Pipeline"""

    def __init__(self):
        self.chunker = get_chunker()
        self.embedding_service = get_embedding_service()
        self.storage = get_qdrant_storage()

    def index_pdf(
        self,
        file_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> DocumentMetadata:
        """
        索引单个 PDF 文件

        Args:
            file_path: PDF 文件路径
            metadata: 可选的元数据（title, industry, report_type 等）

        Returns:
            文档元数据
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        doc_id = str(uuid.uuid4())
        logger.info(f"开始索引: {file_path.name}, doc_id={doc_id}")

        # Step 1: 解析 PDF
        logger.info("Step 1: 解析 PDF...")
        pages = self._parse_pdf(file_path)
        logger.info(f"  解析完成: {len(pages)} 页")

        # Step 2: 分块
        logger.info("Step 2: 文本分块...")
        chunks = self.chunker.chunk_document(doc_id, pages)
        logger.info(f"  分块完成: {len(chunks)} 个分块")

        # Step 3: 生成 Embedding
        logger.info("Step 3: 生成 Embedding...")
        chunks_with_embedding = self._generate_embeddings(chunks)
        logger.info(f"  Embedding 完成")

        # Step 4: 创建元数据
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            file_name=file_path.name,
            file_path=str(file_path),
            title=metadata.get("title", file_path.stem) if metadata else file_path.stem,
            industry=metadata.get("industry") if metadata else None,
            report_type=metadata.get("report_type") if metadata else None,
            total_pages=len(pages),
            total_chunks=len(chunks),
            file_size=file_path.stat().st_size,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Step 5: 存入 Qdrant
        logger.info("Step 4: 存入向量数据库...")
        self.storage.insert_chunks(chunks_with_embedding, doc_metadata)
        logger.info(f"  存储完成")

        logger.info(f"✅ 索引完成: {file_path.name}, {len(chunks)} 个分块")

        return doc_metadata

    def index_directory(
        self,
        directory: str | Path,
        batch_size: int = 10,
        recursive: bool = True,
    ) -> dict:
        """
        批量索引目录下的 PDF 文件

        Args:
            directory: 目录路径
            batch_size: 批处理大小（每处理这么多文件输出一次进度）
            recursive: 是否递归处理子目录

        Returns:
            统计信息
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        # 收集 PDF 文件
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))

        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

        # 确保 collection 存在
        self.storage.create_collection(recreate=False)

        # 预加载模型
        self.embedding_service.load_model()

        # 处理文件
        success = 0
        failed = []

        for i, pdf_file in enumerate(tqdm(pdf_files, desc="索引进度")):
            try:
                self.index_pdf(pdf_file)
                success += 1
            except Exception as e:
                logger.error(f"索引失败: {pdf_file.name} - {e}")
                failed.append({"file": str(pdf_file), "error": str(e)})

        # 重建 BM25 索引
        logger.info("重建 BM25 索引...")
        from src.retrieval import get_bm25_retriever

        bm25 = get_bm25_retriever()
        bm25.build_index(force_rebuild=True)

        return {
            "total": len(pdf_files),
            "success": success,
            "failed": len(failed),
            "failures": failed,
        }

    def _parse_pdf(self, file_path: Path) -> list[dict]:
        """
        解析 PDF 文件

        优先使用 MinerU（如果安装），否则使用 PyMuPDF
        """
        # 尝试使用 MinerU（高质量解析）
        try:
            return self._parse_with_mineru(file_path)
        except ImportError:
            logger.info("MinerU 未安装，使用 PyMuPDF 解析")
        except Exception as e:
            logger.warning(f"MinerU 解析失败，回退到 PyMuPDF: {e}")

        # 回退到 PyMuPDF（轻量快速）
        return self._parse_with_pymupdf(file_path)

    def _parse_with_pymupdf(self, file_path: Path) -> list[dict]:
        """使用 PyMuPDF 解析 PDF（轻量快速）"""
        import fitz  # PyMuPDF

        pages = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                pages.append({"page_number": page_num + 1, "text": text})

        doc.close()
        return pages

    def _parse_with_mineru(self, file_path: Path) -> list[dict]:
        """
        使用 MinerU 解析 PDF（高质量，适合复杂布局）

        需要安装: uv pip install -e ".[mineru]" --prerelease=allow
        """
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.pipe.OCRPipe import OCRPipe
        from magic_pdf.pipe.UNIPipe import UNIPipe
        import json

        logger.info(f"使用 MinerU 解析: {file_path.name}")

        # 读取 PDF
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(str(file_path))

        # 输出目录
        output_dir = settings.parsed_dir / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建 writer
        image_writer = FileBasedDataWriter(str(output_dir / "images"))
        md_writer = FileBasedDataWriter(str(output_dir))

        # 使用 UNIPipe（自动选择 OCR 或文本模式）
        pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)

        # 执行解析
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        # 获取结果
        md_content = pipe.pipe_mk_markdown(str(output_dir / "images"), drop_mode="none")

        # 保存 markdown
        md_writer.write_string(f"{file_path.stem}.md", md_content)

        # 转换为页面格式（MinerU 的 markdown 是整体的，这里简化处理）
        # 实际使用中可以根据 pipe 的中间结果获取页面信息
        pages = [{"page_number": 1, "text": md_content}]

        return pages

    def _generate_embeddings(self, chunks: list) -> list[ChunkWithEmbedding]:
        """为分块生成 Embedding（分批处理，避免内存溢出）"""
        from src.config import settings
        
        batch_size = settings.embedding_batch_size
        total = len(chunks)
        chunks_with_embedding = []
        
        logger.info(f"开始生成 Embedding，共 {total} 个分块，批大小 {batch_size}")

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            # 批量生成 Embedding
            batch_embeddings = self.embedding_service.encode(batch_texts, show_progress=False)
            
            # 组合
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                chunks_with_embedding.append(
                    ChunkWithEmbedding(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        section_title=chunk.section_title,
                        embedding=embedding,
                    )
                )
            
            logger.info(f"  Embedding 进度: {min(i + batch_size, total)}/{total}")

        return chunks_with_embedding


def get_indexing_pipeline() -> IndexingPipeline:
    """获取索引 Pipeline"""
    return IndexingPipeline()

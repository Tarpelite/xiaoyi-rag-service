"""
Qdrant 向量数据库存储层
"""

from typing import Optional

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.models import Chunk, ChunkWithEmbedding, DocumentMetadata


class QdrantStorage:
    """Qdrant 存储管理"""

    _instance: Optional["QdrantStorage"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
        )
        self.collection_name = settings.qdrant_collection
        self._initialized = True

        logger.info(f"Qdrant 连接: {settings.qdrant_url}, collection: {self.collection_name}")

    def is_connected(self) -> bool:
        """检查连接状态"""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant 连接失败: {e}")
            return False

    def create_collection(self, recreate: bool = False) -> bool:
        """
        创建 collection

        Args:
            recreate: 是否重建（会删除已有数据）

        Returns:
            是否创建成功
        """
        try:
            # 检查是否存在
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists and not recreate:
                logger.info(f"Collection {self.collection_name} 已存在")
                return True

            if exists and recreate:
                logger.warning(f"删除已有 collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)

            # 创建 collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
                # 启用 payload 索引（用于过滤）
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )

            # 创建 payload 索引（加速过滤查询）
            self._create_payload_indexes()

            logger.info(f"Collection {self.collection_name} 创建成功")
            return True

        except Exception as e:
            logger.error(f"创建 collection 失败: {e}")
            return False

    def _create_payload_indexes(self):
        """创建 payload 字段索引"""
        index_fields = [
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("file_name", models.PayloadSchemaType.KEYWORD),
            ("industry", models.PayloadSchemaType.KEYWORD),
            ("report_type", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except UnexpectedResponse:
                # 索引可能已存在
                pass

    def insert_chunks(self, chunks: list[ChunkWithEmbedding], metadata: DocumentMetadata) -> int:
        """
        批量插入分块

        Args:
            chunks: 带向量的分块列表
            metadata: 文档元数据

        Returns:
            插入数量
        """
        import uuid as uuid_lib
        
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            # Qdrant point ID 必须是纯 UUID 或整数
            # 为每个 chunk 生成独立的 UUID 作为 point ID
            point_id = str(uuid_lib.uuid4())
            
            point = models.PointStruct(
                id=point_id,
                vector=chunk.embedding,
                payload={
                    # 分块信息
                    "chunk_id": chunk.chunk_id,  # 原始 chunk_id 保存在 payload 里
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "section_title": chunk.section_title,
                    # 文档元数据
                    "file_name": metadata.file_name,
                    "title": metadata.title,
                    "industry": metadata.industry,
                    "report_type": metadata.report_type,
                    "publish_date": metadata.publish_date,
                    "stock_codes": metadata.stock_codes,
                },
            )
            points.append(point)

        # 批量插入
        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"插入 {len(points)} 个分块, doc_id: {metadata.doc_id}")
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        """
        删除文档的所有分块

        Args:
            doc_id: 文档 ID

        Returns:
            删除数量
        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

        logger.info(f"删除文档分块: doc_id={doc_id}")
        return result.status

    def search_vector(
        self,
        query_vector: list[float],
        top_k: int = 100,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            filters: 过滤条件

        Returns:
            检索结果列表
        """
        # 构建过滤条件
        qdrant_filter = self._build_filter(filters) if filters else None

        # 新版 qdrant-client 使用 query_points
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": r.payload.get("chunk_id"),
                "doc_id": r.payload.get("doc_id"),
                "content": r.payload.get("content"),
                "page_number": r.payload.get("page_number"),
                "file_name": r.payload.get("file_name"),
                "title": r.payload.get("title"),
                "section_title": r.payload.get("section_title"),
                "score": r.score,
            }
            for r in results
        ]

    def _build_filter(self, filters: dict) -> models.Filter:
        """构建 Qdrant 过滤条件"""
        conditions = []

        if filters.get("doc_ids"):
            conditions.append(
                models.FieldCondition(
                    key="doc_id", match=models.MatchAny(any=filters["doc_ids"])
                )
            )

        if filters.get("industries"):
            conditions.append(
                models.FieldCondition(
                    key="industry", match=models.MatchAny(any=filters["industries"])
                )
            )

        if filters.get("report_types"):
            conditions.append(
                models.FieldCondition(
                    key="report_type", match=models.MatchAny(any=filters["report_types"])
                )
            )

        if filters.get("stock_codes"):
            # stock_codes 是数组，使用 MatchAny
            for code in filters["stock_codes"]:
                conditions.append(
                    models.FieldCondition(
                        key="stock_codes", match=models.MatchValue(value=code)
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    def get_collection_info(self) -> dict:
        """获取 collection 统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.name,
            }
        except Exception as e:
            logger.error(f"获取 collection 信息失败: {e}")
            return {}

    def get_all_chunks_for_bm25(self) -> list[dict]:
        """
        获取所有分块用于构建 BM25 索引

        注意：这个方法会加载全部数据到内存，大规模数据时需要分页
        """
        all_chunks = []
        offset = None
        limit = 1000

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for r in results:
                all_chunks.append(
                    {
                        "chunk_id": r.payload.get("chunk_id"),
                        "doc_id": r.payload.get("doc_id"),
                        "content": r.payload.get("content"),
                        "page_number": r.payload.get("page_number"),
                        "file_name": r.payload.get("file_name"),
                        "title": r.payload.get("title"),
                    }
                )

            if offset is None:
                break

        return all_chunks


def get_qdrant_storage() -> QdrantStorage:
    """获取 Qdrant 存储单例"""
    return QdrantStorage()
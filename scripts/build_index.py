#!/usr/bin/env python
"""
批量构建索引脚本
================

使用方法：
    python scripts/build_index.py --input data/pdfs
    python scripts/build_index.py --input data/pdfs --batch-size 10 --recreate
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.indexing import get_indexing_pipeline
from src.storage import get_qdrant_storage


def main():
    parser = argparse.ArgumentParser(description="批量构建研报索引")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(settings.pdf_dir),
        help="PDF 文件目录",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="批处理大小",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="是否重建索引（删除已有数据）",
    )

    parser.add_argument(
        "--single",
        "-s",
        type=str,
        help="索引单个文件",
    )

    args = parser.parse_args()

    # 初始化存储
    storage = get_qdrant_storage()

    if not storage.is_connected():
        logger.error("无法连接到 Qdrant，请确保 Qdrant 服务已启动")
        logger.error(f"尝试连接: {settings.qdrant_host}:{settings.qdrant_port}")
        sys.exit(1)

    # 创建/重建 collection
    if args.recreate:
        logger.warning("⚠️ 将删除已有索引并重建！")
        input("按 Enter 继续，Ctrl+C 取消...")

    storage.create_collection(recreate=args.recreate)

    # 获取 pipeline
    pipeline = get_indexing_pipeline()

    # 索引
    if args.single:
        # 索引单个文件
        file_path = Path(args.single)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            sys.exit(1)

        result = pipeline.index_pdf(file_path)
        logger.info(f"✅ 索引完成: {result.file_name}, {result.total_chunks} 个分块")

    else:
        # 批量索引目录
        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"目录不存在: {input_dir}")
            sys.exit(1)

        result = pipeline.index_directory(input_dir, batch_size=args.batch_size)

        logger.info("=" * 50)
        logger.info(f"索引完成!")
        logger.info(f"  总文件数: {result['total']}")
        logger.info(f"  成功: {result['success']}")
        logger.info(f"  失败: {result['failed']}")

        if result["failures"]:
            logger.warning("失败文件列表:")
            for f in result["failures"]:
                logger.warning(f"  - {f['file']}: {f['error']}")


if __name__ == "__main__":
    main()

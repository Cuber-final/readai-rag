#!/usr/bin/env python
"""
测试RAG功能的脚本
使用此脚本测试文档加载、向量索引创建和检索问答功能
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from readai.services.document_loader import document_loader
from readai.services.vector_store import vector_store_service
from readai.services.rag_engine import rag_engine
from readai.services.embedding import embedding_service
from readai.db.models import Book
from readai.core.config import settings, ensure_directories


async def test_rag(file_path: str, query: str):
    """测试RAG流程"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # 确保目录存在
        ensure_directories()
        
        # 提取文件信息
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in [".pdf", ".epub"]:
            logger.error(f"不支持的文件类型: {file_ext}，仅支持PDF和EPUB格式")
            return
        
        # 提取元数据
        metadata = document_loader.extract_metadata(file_path)
        
        # 创建临时Book对象
        book = Book(
            id="test_book",
            title=metadata.title,
            author=metadata.author,
            file_path=file_path,
            file_type=metadata.file_type
        )
        
        # 处理文档
        logger.info(f"开始处理文档: {file_path}")
        success = await rag_engine.process_book(book)
        
        if not success:
            logger.error("文档处理失败")
            return
        
        # 测试检索和问答
        logger.info(f"提问: {query}")
        logger.info("RAG回答:")
        
        # 使用非流式回答测试
        response, sources = rag_engine.chat_with_book_no_stream("test_book", query)
        
        print("\n" + "=" * 50)
        print(f"问题: {query}")
        print("-" * 50)
        print(f"回答: {response}")
        print("-" * 50)
        print("参考来源:")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source['text']}")
        print("=" * 50)
        
        # 清理测试数据
        vector_store_service.delete_index("test_book")
        logger.info("测试完成，已清理临时数据")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试RAG功能")
    parser.add_argument("file_path", help="文件路径，支持PDF和EPUB格式")
    parser.add_argument("query", help="测试查询问题")
    args = parser.parse_args()
    
    asyncio.run(test_rag(args.file_path, args.query)) 
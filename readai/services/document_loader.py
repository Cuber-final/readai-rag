import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import uuid

from llama_index.core import Document
from llama_index.readers.file import PyPDFReader, EpubReader
from llama_index.core.node_parser import SentenceSplitter

from readai.core.config import settings
from readai.core.exceptions import DocumentLoadException
from readai.db.models import Book
from readai.core.models import BookMetadata

logger = logging.getLogger(__name__)


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self):
        self.pdf_reader = PyPDFReader()
        self.epub_reader = EpubReader()
        self.node_parser = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == ".pdf":
                logger.info(f"加载PDF文档: {file_path}")
                documents = self.pdf_reader.load_data(file_path)
            elif file_ext in [".epub"]:
                logger.info(f"加载EPUB文档: {file_path}")
                documents = self.epub_reader.load_data(file_path)
            else:
                raise DocumentLoadException(file_path, f"不支持的文件类型: {file_ext}")
            
            logger.info(f"文档加载成功, 总字符数: {sum(len(doc.text) for doc in documents)}")
            return documents
        except Exception as e:
            logger.error(f"文档加载失败: {str(e)}")
            raise DocumentLoadException(file_path, str(e))
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分块"""
        try:
            logger.info("开始分割文档...")
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"文档分割完成, 总块数: {len(nodes)}")
            
            # 将节点转换回Document对象
            split_docs = []
            for node in nodes:
                doc = Document(
                    text=node.text,
                    metadata=node.metadata,
                    id_=node.id_,
                )
                split_docs.append(doc)
            
            return split_docs
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            raise DocumentLoadException("", f"文档分割失败: {str(e)}")
    
    def extract_metadata(self, file_path: str) -> BookMetadata:
        """从文件中提取元数据"""
        try:
            # 获取文件信息
            file_path = str(Path(file_path).absolute())
            file_name = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()
            
            # 生成书籍ID
            book_id = str(uuid.uuid4())
            
            # 创建元数据
            metadata = BookMetadata(
                book_id=book_id,
                title=file_name.replace(file_ext, ""),  # 使用文件名作为标题
                file_path=file_path,
                file_type=file_ext.replace(".", ""),
                processed=False
            )
            
            # 尝试从文件中提取更多元数据
            try:
                if file_ext == ".pdf":
                    # 从PDF提取元数据
                    docs = self.pdf_reader.load_data(file_path)
                    if docs and hasattr(docs[0], "metadata"):
                        pdf_meta = docs[0].metadata
                        if "title" in pdf_meta and pdf_meta["title"]:
                            metadata.title = pdf_meta["title"]
                        if "author" in pdf_meta and pdf_meta["author"]:
                            metadata.author = pdf_meta["author"]
                
                elif file_ext == ".epub":
                    # 从EPUB提取元数据
                    docs = self.epub_reader.load_data(file_path)
                    if docs and hasattr(docs[0], "metadata"):
                        epub_meta = docs[0].metadata
                        if "title" in epub_meta and epub_meta["title"]:
                            metadata.title = epub_meta["title"]
                        if "creator" in epub_meta and epub_meta["creator"]:
                            metadata.author = epub_meta["creator"]
                        if "publisher" in epub_meta and epub_meta["publisher"]:
                            metadata.publisher = epub_meta["publisher"]
                        if "language" in epub_meta and epub_meta["language"]:
                            metadata.language = epub_meta["language"]
            except Exception as e:
                logger.warning(f"提取元数据时出现警告 (继续处理): {str(e)}")
            
            return metadata
        except Exception as e:
            logger.error(f"提取元数据失败: {str(e)}")
            raise DocumentLoadException(file_path, f"提取元数据失败: {str(e)}")


# 创建文档加载器实例
document_loader = DocumentLoader() 
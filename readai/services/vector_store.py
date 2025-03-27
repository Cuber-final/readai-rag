import logging
import os
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from readai.components.embedding import embedding_service
from readai.core.config import settings
from readai.core.exceptions import VectorStoreException

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量存储服务"""

    def __init__(self):
        self.embed_model = embedding_service.embed_model
        self.vector_store_type = settings.VECTOR_STORE_TYPE

    def _get_chroma_client(self, collection_name: str) -> chromadb.PersistentClient:
        """获取ChromaDB客户端"""
        try:
            persist_dir = os.path.join(settings.CHROMA_PERSIST_DIR, collection_name)
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=persist_dir)
        except Exception as e:
            logger.error(f"创建ChromaDB客户端失败: {e!s}")
            raise VectorStoreException(f"创建ChromaDB客户端失败: {e!s}")

    def _get_qdrant_client(self) -> QdrantClient:
        """获取Qdrant客户端"""
        try:
            return QdrantClient(url=settings.QDRANT_URL)
        except Exception as e:
            logger.error(f"创建Qdrant客户端失败: {e!s}")
            raise VectorStoreException(f"创建Qdrant客户端失败: {e!s}")

    def create_or_update_index(
        self, book_id: str, documents: list[Document]
    ) -> VectorStoreIndex:
        """创建或更新索引"""
        try:
            if self.vector_store_type == "chroma":
                return self._create_or_update_chroma_index(book_id, documents)
            elif self.vector_store_type == "qdrant":
                return self._create_or_update_qdrant_index(book_id, documents)
            else:
                raise VectorStoreException(
                    f"不支持的向量存储类型: {self.vector_store_type}"
                )
        except Exception as e:
            logger.error(f"创建或更新索引失败: {e!s}")
            raise VectorStoreException(f"创建或更新索引失败: {e!s}")

    def _create_or_update_chroma_index(
        self, book_id: str, documents: list[Document]
    ) -> VectorStoreIndex:
        """创建或更新ChromaDB索引"""
        try:
            # 获取ChromaDB客户端
            chroma_client = self._get_chroma_client(book_id)

            # 创建或获取集合
            collection = chroma_client.get_or_create_collection(book_id)

            # 创建向量存储
            vector_store = ChromaVectorStore(chroma_collection=collection)

            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 创建索引
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embed_model
            )

            logger.info(
                f"成功创建或更新ChromaDB索引: {book_id}, 文档数量: {len(documents)}"
            )
            return index
        except Exception as e:
            logger.error(f"创建或更新ChromaDB索引失败: {e!s}")
            raise VectorStoreException(f"创建或更新ChromaDB索引失败: {e!s}")

    def _create_or_update_qdrant_index(
        self, book_id: str, documents: list[Document]
    ) -> VectorStoreIndex:
        """创建或更新Qdrant索引"""
        try:
            # 获取Qdrant客户端
            client = self._get_qdrant_client()

            # 获取向量维度
            embedding_dimension = embedding_service.get_embedding_dimension()

            # 创建或更新集合
            try:
                client.get_collection(book_id)
                # 集合已存在，删除旧的集合
                client.delete_collection(book_id)
                logger.info(f"删除现有Qdrant集合: {book_id}")
            except Exception:
                # 集合不存在，忽略错误
                pass

            # 创建新集合
            client.create_collection(
                collection_name=book_id,
                vectors_config=qdrant_models.VectorParams(
                    size=embedding_dimension, distance=qdrant_models.Distance.COSINE
                ),
            )

            # 创建向量存储
            vector_store = QdrantVectorStore(client=client, collection_name=book_id)

            # 创建存储上下文
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 创建索引
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embed_model
            )

            logger.info(
                f"成功创建或更新Qdrant索引: {book_id}, 文档数量: {len(documents)}"
            )
            return index
        except Exception as e:
            logger.error(f"创建或更新Qdrant索引失败: {e!s}")
            raise VectorStoreException(f"创建或更新Qdrant索引失败: {e!s}")

    def get_index(self, book_id: str) -> VectorStoreIndex | None:
        """获取索引"""
        try:
            if self.vector_store_type == "chroma":
                return self._get_chroma_index(book_id)
            elif self.vector_store_type == "qdrant":
                return self._get_qdrant_index(book_id)
            else:
                raise VectorStoreException(
                    f"不支持的向量存储类型: {self.vector_store_type}"
                )
        except Exception as e:
            logger.error(f"获取索引失败: {e!s}")
            return None

    def _get_chroma_index(self, book_id: str) -> VectorStoreIndex | None:
        """获取ChromaDB索引"""
        try:
            # 获取ChromaDB客户端
            chroma_client = self._get_chroma_client(book_id)

            # 检查集合是否存在
            try:
                collection = chroma_client.get_collection(book_id)
            except Exception:
                logger.warning(f"ChromaDB集合不存在: {book_id}")
                return None

            # 创建向量存储
            vector_store = ChromaVectorStore(chroma_collection=collection)

            # 创建索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
            )

            logger.info(f"成功获取ChromaDB索引: {book_id}")
            return index
        except Exception as e:
            logger.error(f"获取ChromaDB索引失败: {e!s}")
            return None

    def _get_qdrant_index(self, book_id: str) -> VectorStoreIndex | None:
        """获取Qdrant索引"""
        try:
            # 获取Qdrant客户端
            client = self._get_qdrant_client()

            # 检查集合是否存在
            try:
                client.get_collection(book_id)
            except Exception:
                logger.warning(f"Qdrant集合不存在: {book_id}")
                return None

            # 创建向量存储
            vector_store = QdrantVectorStore(client=client, collection_name=book_id)

            # 创建索引
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
            )

            logger.info(f"成功获取Qdrant索引: {book_id}")
            return index
        except Exception as e:
            logger.error(f"获取Qdrant索引失败: {e!s}")
            return None

    def delete_index(self, book_id: str) -> bool:
        """删除索引"""
        try:
            if self.vector_store_type == "chroma":
                return self._delete_chroma_index(book_id)
            elif self.vector_store_type == "qdrant":
                return self._delete_qdrant_index(book_id)
            else:
                raise VectorStoreException(
                    f"不支持的向量存储类型: {self.vector_store_type}"
                )
        except Exception as e:
            logger.error(f"删除索引失败: {e!s}")
            return False

    def _delete_chroma_index(self, book_id: str) -> bool:
        """删除ChromaDB索引"""
        try:
            # 获取ChromaDB客户端
            chroma_client = self._get_chroma_client(book_id)

            # 删除集合
            try:
                chroma_client.delete_collection(book_id)
                logger.info(f"成功删除ChromaDB集合: {book_id}")
                return True
            except Exception as e:
                logger.warning(f"删除ChromaDB集合失败(可能不存在): {e!s}")
                return False
        except Exception as e:
            logger.error(f"删除ChromaDB索引失败: {e!s}")
            return False

    def _delete_qdrant_index(self, book_id: str) -> bool:
        """删除Qdrant索引"""
        try:
            # 获取Qdrant客户端
            client = self._get_qdrant_client()

            # 删除集合
            try:
                client.delete_collection(book_id)
                logger.info(f"成功删除Qdrant集合: {book_id}")
                return True
            except Exception as e:
                logger.warning(f"删除Qdrant集合失败(可能不存在): {e!s}")
                return False
        except Exception as e:
            logger.error(f"删除Qdrant索引失败: {e!s}")
            return False

    def check_index_exists(self, book_id: str) -> bool:
        """检查索引是否存在"""
        try:
            if self.vector_store_type == "chroma":
                # 获取ChromaDB客户端
                chroma_client = self._get_chroma_client(book_id)

                # 检查集合是否存在
                try:
                    chroma_client.get_collection(book_id)
                    return True
                except Exception:
                    return False

            elif self.vector_store_type == "qdrant":
                # 获取Qdrant客户端
                client = self._get_qdrant_client()

                # 检查集合是否存在
                try:
                    client.get_collection(book_id)
                    return True
                except Exception:
                    return False

            else:
                raise VectorStoreException(
                    f"不支持的向量存储类型: {self.vector_store_type}"
                )
        except Exception as e:
            logger.error(f"检查索引是否存在失败: {e!s}")
            return False


# 创建向量存储服务实例
vector_store_service = VectorStoreService()

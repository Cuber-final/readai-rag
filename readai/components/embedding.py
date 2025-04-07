import logging

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from readai.core.config import settings
from readai.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding服务类"""

    def __init__(self):
        self.embed_model = self._get_embedding_model()

    def _get_embedding_model(self) -> BaseEmbedding:
        """获取Embedding模型"""
        try:
            if settings.EMBEDDING_MODEL_TYPE == "huggingface":
                logger.info(
                    f"使用HuggingFace Embedding模型: {settings.EMBEDDING_MODEL_NAME}"
                )
                return HuggingFaceEmbedding(
                    model_name=settings.EMBEDDING_MODEL_NAME, embed_batch_size=8
                )
            elif settings.EMBEDDING_MODEL_TYPE == "ollama":
                logger.info(
                    f"使用Ollama Embedding模型: {settings.EMBEDDING_MODEL_NAME}"
                )
                return OllamaEmbedding(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    base_url=settings.OLLAMA_BASE_URL,
                )
            else:
                raise EmbeddingException(
                    f"不支持的Embedding模型类型: {settings.EMBEDDING_MODEL_TYPE}"
                )
        except Exception as e:
            logger.error(f"加载Embedding模型失败: {e!s}")
            raise EmbeddingException(f"加载Embedding模型失败: {e!s}") from e

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取文本的Embedding"""
        try:
            embeddings = []
            for text in texts:
                embedding = self.embed_model.get_text_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"获取Embedding失败: {e!s}")
            raise EmbeddingException(f"获取Embedding失败: {e!s}") from e

    def get_embedding(self, text: str) -> list[float]:
        """获取单个文本的Embedding"""
        try:
            return self.embed_model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"获取Embedding失败: {e!s}")
            raise EmbeddingException(f"获取Embedding失败: {e!s}") from e

    def get_embedding_dimension(self) -> int:
        """获取Embedding维度"""
        try:
            # 使用一个简单的文本获取Embedding并计算维度
            embedding = self.get_embedding("测试文本")
            return len(embedding)
        except Exception as e:
            logger.error(f"获取Embedding维度失败: {e!s}")
            raise EmbeddingException(f"获取Embedding维度失败: {e!s}") from e


# 创建全局Embedding服务实例
embedding_service = EmbeddingService()

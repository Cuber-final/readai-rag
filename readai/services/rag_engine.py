import logging
from collections.abc import Generator
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

from readai.components.document_loader import document_loader
from readai.components.embedding import embedding_service
from readai.components.llm import llm_service
from readai.core.exceptions import BookNotProcessedException, VectorStoreException
from readai.db.models import Book
from readai.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG引擎服务"""

    def __init__(self):
        self.vector_store_service = vector_store_service
        self.document_loader = document_loader
        self.llm_service = llm_service
        self.embed_model = embedding_service.embed_model

    async def process_book(self, book: Book) -> bool:
        """处理书籍，生成向量索引"""
        try:
            logger.info(f"开始处理书籍: {book.id}, 路径: {book.file_path}")

            # 加载文档
            documents = self.document_loader.load_document(book.file_path)

            # 分割文档
            split_documents = self.document_loader.split_documents(documents)

            # 为每个文档块添加元数据
            for doc in split_documents:
                doc.metadata.update(
                    {
                        "book_id": book.id,
                        "book_title": book.title,
                        "book_author": book.author,
                    }
                )

            # 创建索引
            index = self.vector_store_service.create_or_update_index(
                book.id, split_documents
            )

            logger.info(f"书籍处理完成: {book.id}, 总块数: {len(split_documents)}")
            return True
        except Exception as e:
            logger.error(f"处理书籍失败: {e!s}")
            return False

    def get_index(self, book_id: str) -> VectorStoreIndex:
        """获取书籍索引"""
        index = self.vector_store_service.get_index(book_id)
        if not index:
            raise BookNotProcessedException(book_id)
        return index

    def chat_with_book(
        self,
        book_id: str,
        message: str,
        history: list[dict[str, str]] = None,
        model_name: str | None = None,
    ) -> Generator[str, None, None]:
        """与书籍聊天（流式返回）"""
        try:
            # 检查书籍索引是否存在
            index = self.get_index(book_id)

            # 获取上下文信息
            contexts = self._get_relevant_contexts(index, message, top_k=3)
            contexts_text = "\n\n".join([node.text for node in contexts])

            # 构建包含检索上下文的系统提示
            system_prompt = f"""你是一个专业的电子书阅读助手，负责回答用户有关书籍内容的问题。
请基于以下从书中提取的相关段落回答用户的问题：

{contexts_text}

回答时注意以下几点：
1. 只回答与提供内容相关的部分，如果无法从上下文中找到答案，请诚实地说明你不知道。
2. 不要编造内容或使用你自己的知识回答。
3. 使用专业、简洁、有条理的语言回答问题。
4. 引用原文时可以适当调整语序使回答更流畅，但不要改变原意。
5. 如果问题不合适或无法回答，请引导用户询问更多关于书籍内容的问题。
"""

            # 准备聊天消息
            messages = []

            # 添加系统消息
            messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                for msg in history:
                    messages.append(msg)

            # 添加当前用户消息
            messages.append({"role": "user", "content": message})

            # 进行流式对话
            return self.llm_service.chat_completion_stream(
                messages, model_name=model_name
            )

        except BookNotProcessedException as e:
            logger.error(f"书籍未处理: {e}")
            yield "错误: 该书籍尚未处理为向量形式，请先处理该书籍。"
        except Exception as e:
            logger.error(f"与书籍聊天失败: {e!s}")
            yield f"与书籍聊天失败: {e!s}"

    def chat_with_book_no_stream(
        self,
        book_id: str,
        message: str,
        history: list[dict[str, str]] = None,
        model_name: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """与书籍聊天（非流式返回）"""
        try:
            # 检查书籍索引是否存在
            index = self.get_index(book_id)

            # 获取上下文信息
            contexts = self._get_relevant_contexts(index, message, top_k=3)
            contexts_text = "\n\n".join([node.text for node in contexts])

            # 提取源文档信息用于引用
            sources = []
            for node in contexts:
                source_info = {
                    "text": node.text[:100] + "..."
                    if len(node.text) > 100
                    else node.text,
                    "metadata": node.metadata,
                    "similarity": node.score if hasattr(node, "score") else None,
                }
                sources.append(source_info)

            # 构建包含检索上下文的系统提示
            system_prompt = f"""你是一个专业的电子书阅读助手，负责回答用户有关书籍内容的问题。
请基于以下从书中提取的相关段落回答用户的问题：

{contexts_text}

回答时注意以下几点：
1. 只回答与提供内容相关的部分，如果无法从上下文中找到答案，请诚实地说明你不知道。
2. 不要编造内容或使用你自己的知识回答。
3. 使用专业、简洁、有条理的语言回答问题。
4. 引用原文时可以适当调整语序使回答更流畅，但不要改变原意。
5. 如果问题不合适或无法回答，请引导用户询问更多关于书籍内容的问题。
"""

            # 准备聊天消息
            messages = []

            # 添加系统消息
            messages.append({"role": "system", "content": system_prompt})

            # 添加历史消息
            if history:
                for msg in history:
                    messages.append(msg)

            # 添加当前用户消息
            messages.append({"role": "user", "content": message})

            # 进行非流式对话
            response = self.llm_service.chat_completion(messages, model_name=model_name)

            return response, sources

        except BookNotProcessedException as e:
            logger.error(f"书籍未处理: {e}")
            return "错误: 该书籍尚未处理为向量形式，请先处理该书籍。", []
        except Exception as e:
            logger.error(f"与书籍聊天失败: {e!s}")
            return f"与书籍聊天失败: {e!s}", []

    def _get_relevant_contexts(
        self, index: VectorStoreIndex, query: str, top_k: int = 3
    ) -> list[NodeWithScore]:
        """获取与查询相关的上下文"""
        try:
            # 创建检索器
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k,
            )

            # 检索相关节点
            nodes = retriever.retrieve(query)

            # 使用相似度后处理器进行过滤
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
            nodes = postprocessor.postprocess_nodes(nodes)

            logger.info(f"检索到{len(nodes)}个相关节点")
            return nodes
        except Exception as e:
            logger.error(f"获取相关上下文失败: {e!s}")
            raise VectorStoreException(f"获取相关上下文失败: {e!s}")


# 创建全局RAG引擎实例
rag_engine = RAGEngine()

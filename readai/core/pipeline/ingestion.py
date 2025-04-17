# 该文件主要实现以下功能
# 1. 调用已经实现的方法将epub文件并处理为markdown后
# 2、langchain的UnstructuredMarkdownLoader组件加载并生成llamaindex需要的document对象以及nodes
# 3. 利用SimpleNodeParser，small2big方式处理生成多层级节点作为存储到qdrant向量库前的准备
# 4. 数据持久化，SimpleDocumentStore,以及qdrant的向量存储来实现
# 5. 根据collection_name以及persist_dir作为参数，获取已有StorageContext返回用于检索
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from readai.components.embedding import embedding_service
from readai.components.epub2md_loader import preprocess_book

load_dotenv()

# 获取项目根目录路径
project_root = Path(os.getenv("PROJECT_ROOT"))
logger.add(f"{project_root}/logs/ingestion.log", rotation="10 MB")


class IngestionError(Exception):
    """数据摄入过程中的自定义异常"""

    pass


class DataIngestionPipeline:
    """数据摄入管道，处理从epub到向量数据库的全过程"""

    def __init__(
        self,
        qdrant_url="http://localhost:6333",
        sub_chunk_sizes=None,
        embed_model=None,
        llm=None,
    ):
        """初始化数据摄入管道

        Args:
            qdrant_url: Qdrant服务地址
            sub_chunk_sizes: small2big方式的多层级切分大小
            embed_model: 嵌入模型
            llm: 语言模型
        """
        self.qdrant_url = qdrant_url
        try:
            self.client = QdrantClient(url=qdrant_url)
            self.aclient = AsyncQdrantClient(url=qdrant_url)
        except Exception as e:
            logger.error(f"无法连接到Qdrant服务 {qdrant_url}: {e!s}")
            raise IngestionError(f"Qdrant连接失败: {e!s}")

        self.sub_chunk_sizes = sub_chunk_sizes or [128, 512, 2048]
        self.embed_model = embed_model
        self.llm = llm

    def process_epub_to_markdown(self, book_path: str) -> tuple[dict, str]:
        """处理epub文件为markdown格式

        Args:
            book_path: epub文件路径

        Returns:
            元数据和生成的markdown文件路径

        Raises:
            IngestionError: 处理epub文件失败时抛出
        """
        logger.info(f"开始处理epub文件: {book_path}")

        # 检查文件是否存在
        if not os.path.exists(book_path):
            error_msg = f"文件不存在: {book_path}"
            logger.error(error_msg)
            raise IngestionError(error_msg)

        # 检查文件格式
        if not book_path.lower().endswith(".epub"):
            error_msg = f"文件格式不支持，需要epub格式: {book_path}"
            logger.error(error_msg)
            raise IngestionError(error_msg)

        try:
            metadata, cleaned_md_path = preprocess_book(book_path)
            logger.info(f"epub文件处理完成: {cleaned_md_path}")
            return metadata, cleaned_md_path
        except Exception as e:
            error_msg = f"处理epub文件失败: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise IngestionError(error_msg) from e

    def load_markdown_to_documents(self, markdown_path: str) -> list[Document]:
        """将markdown文件加载为Document对象

        Args:
            markdown_path: markdown文件路径

        Returns:
            LlamaIndex Document对象列表

        Raises:
            IngestionError: 加载markdown文件失败时抛出
        """
        logger.info(f"开始加载markdown文件: {markdown_path}")

        # 检查文件是否存在
        if not os.path.exists(markdown_path):
            error_msg = f"markdown文件不存在: {markdown_path}"
            logger.error(error_msg)
            raise IngestionError(error_msg)

        try:
            # 使用UnstructuredMarkdownLoader加载
            loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
            langchain_docs = loader.load()

            # 过滤不必要的metadata信息
            for doc in langchain_docs:
                category = doc.metadata.get("category", "")
                filename = doc.metadata.get("filename", "")
                doc.metadata = {
                    "category": category,
                    "filename": filename,
                }

            # 转换为LlamaIndex文档对象
            llamaindex_docs = [
                Document(
                    text=doc.page_content,
                    metadata=doc.metadata,
                    metadata_seperator="::",
                    metadata_template="{key}=>{value}",
                    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                )
                for doc in langchain_docs
            ]

            if not llamaindex_docs:
                logger.warning(
                    f"markdown文件加载完成，但未找到有效文档: {markdown_path}"
                )
            else:
                logger.info(f"markdown文件加载完成，共{len(llamaindex_docs)}个文档")

            return llamaindex_docs

        except Exception as e:
            error_msg = f"加载markdown文件失败: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise IngestionError(error_msg) from e

    def create_nodes_with_small2big(
        self, documents: list[Document]
    ) -> tuple[list, list]:
        """使用small2big方式处理文档创建多层级节点

        Args:
            documents: LlamaIndex Document对象列表

        Returns:
            基础节点列表和所有节点列表

        Raises:
            IngestionError: 创建节点失败时抛出
        """
        logger.info("开始创建多层级节点")

        if not documents:
            error_msg = "没有文档可供处理"
            logger.error(error_msg)
            raise IngestionError(error_msg)

        try:
            # 创建基础节点解析器
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=50,
                separator="\n\n",
                secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
            )

            # 获取基础节点
            base_nodes = node_parser.get_nodes_from_documents(
                documents, show_progress=True
            )

            if not base_nodes:
                logger.warning("未能从文档中生成任何基础节点")
                return [], []

            logger.info(f"创建基础节点完成，共{len(base_nodes)}个节点")

            # 创建子节点解析器
            sub_node_parsers = [
                SimpleNodeParser.from_defaults(
                    chunk_size=c,
                    chunk_overlap=50,
                    separator="\n\n",
                    secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
                )
                for c in self.sub_chunk_sizes
            ]

            # 使用small2big方式创建多层级节点
            all_nodes = []
            for base_node in base_nodes:
                for parser in sub_node_parsers:
                    sub_nodes = parser.get_nodes_from_documents([base_node])
                    sub_index_nodes = [
                        IndexNode.from_text_node(sn, base_node.node_id)
                        for sn in sub_nodes
                    ]
                    all_nodes.extend(sub_index_nodes)

                # 将原始节点也添加到节点集合中
                original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                all_nodes.append(original_node)

            logger.info(f"创建多层级节点完成，共{len(all_nodes)}个节点")
            return base_nodes, all_nodes

        except Exception as e:
            error_msg = f"创建多层级节点失败: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise IngestionError(error_msg) from e

    def persist_to_qdrant(
        self, all_nodes: list, collection_name: str, persist_dir: str = None
    ) -> VectorStoreIndex | None:
        """将节点持久化到Qdrant向量数据库

        Args:
            all_nodes: 所有节点列表
            collection_name: 集合名称
            persist_dir: 本地持久化目录

        Returns:
            向量索引对象或None（如果失败）

        Raises:
            IngestionError: 持久化到Qdrant失败时抛出
        """
        logger.info(f"开始持久化到Qdrant，集合名称: {collection_name}")

        if not all_nodes:
            error_msg = "没有节点可供持久化"
            logger.error(error_msg)
            raise IngestionError(error_msg)

        if not self.embed_model:
            logger.warning("未提供嵌入模型，这可能导致索引创建失败")

        # 设置持久化目录
        if persist_dir is None:
            persist_dir = project_root / f"data/indices/{collection_name}"

        try:
            # 创建持久化目录（如果不存在）
            os.makedirs(persist_dir, exist_ok=True)

            # 创建文档存储
            docstore = SimpleDocumentStore()
            docstore.add_documents(all_nodes)

            # 创建向量存储
            vector_store = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name=collection_name,
            )

            # 检查集合是否存在
            try:
                collection_exists = self.client.collection_exists(collection_name)
            except UnexpectedResponse as e:
                logger.error(f"检查Qdrant集合时出错: {e!s}")
                raise IngestionError(f"Qdrant服务错误: {e!s}") from e

            if collection_exists:
                logger.info(f"集合 '{collection_name}' 已存在，加载现有索引")
                try:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=persist_dir, vector_store=vector_store
                    )
                    index = VectorStoreIndex.from_storage_context(
                        storage_context=storage_context,
                        embed_model=self.embed_model,
                        llm=self.llm,
                    )
                except Exception as e:
                    error_msg = f"加载现有索引失败: {e!s}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    raise IngestionError(error_msg) from e
            else:
                logger.info(f"集合 '{collection_name}' 不存在，创建新索引")
                try:
                    storage_context = StorageContext.from_defaults(
                        vector_store=vector_store,
                        docstore=docstore,
                    )
                    index = VectorStoreIndex(
                        nodes=all_nodes,
                        storage_context=storage_context,
                        embed_model=self.embed_model,
                        llm=self.llm,
                        show_progress=True,
                    )

                    # 持久化到本地
                    try:
                        storage_context.persist(persist_dir)
                        logger.info(f"索引已持久化到: {persist_dir}")
                    except Exception as e:
                        logger.error(f"持久化索引失败: {e!s}\n{traceback.format_exc()}")
                        # 这里我们不抛出异常，因为索引已经创建成功，只是持久化失败

                except Exception as e:
                    error_msg = f"创建新索引失败: {e!s}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    raise IngestionError(error_msg) from e

            return index

        except IngestionError:
            # 直接重新抛出已处理的IngestionError
            raise
        except Exception as e:
            error_msg = f"持久化到Qdrant失败: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise IngestionError(error_msg) from e

    def run_pipeline(
        self, book_path: str, collection_name: str = None, persist_dir: str = None
    ) -> VectorStoreIndex | None:
        """运行完整的数据摄入管道

        Args:
            book_path: epub文件路径
            collection_name: 可选，集合名称
            persist_dir: 可选，持久化目录

        Returns:
            向量索引对象或None（如果失败）
        """
        logger.info(f"开始运行数据摄入管道，处理书籍: {book_path}")

        try:
            # 如果没有指定集合名称，使用文件名作为集合名称
            if collection_name is None:
                collection_name = Path(book_path).stem
                logger.info(f"未指定集合名称，使用文件名: {collection_name}")

            # 1. 处理epub为markdown
            metadata, md_path = self.process_epub_to_markdown(book_path)
            if not md_path or not os.path.exists(md_path):
                error_msg = f"markdown文件生成失败或不存在: {md_path}"
                logger.error(error_msg)
                raise IngestionError(error_msg)

            # 2. 加载markdown为documents
            documents = self.load_markdown_to_documents(md_path)
            if not documents:
                error_msg = f"从markdown加载文档失败，未获取到任何文档: {md_path}"
                logger.error(error_msg)
                raise IngestionError(error_msg)

            # 3. 创建多层级节点
            base_nodes, all_nodes = self.create_nodes_with_small2big(documents)
            if not all_nodes:
                error_msg = "创建节点失败，未生成任何节点"
                logger.error(error_msg)
                raise IngestionError(error_msg)

            # 4. 数据持久化
            index = self.persist_to_qdrant(all_nodes, collection_name, persist_dir)
            logger.info(f"数据摄入管道运行完成，成功处理书籍: {book_path}")
            return index

        except IngestionError as e:
            # 已经记录过日志的错误，这里再记录一个总结性的错误
            logger.error(f"数据摄入管道运行失败: {e!s}")
            return None
        except Exception as e:
            # 未预期的错误
            error_msg = f"数据摄入管道运行过程中发生未知错误: {e!s}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None


def load_index(collection_name: str, persist_dir: str = None) -> BaseIndex | None:
    """加载本地索引

    Args:
        collection_name: 集合名称
        persist_dir: 本地持久化目录
    """
    from llama_index.core.indices import load_index_from_storage

    persist_dir = project_root / "data/qdrant" / collection_name

    # 创建vector_store
    qdrant_vs = QdrantVectorStore(
        client=QdrantClient(url="http://localhost:6333"),
        aclient=AsyncQdrantClient(url="http://localhost:6333"),
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=qdrant_vs
    )
    loaded_index = load_index_from_storage(
        storage_context, embed_model=embedding_service.get_embed_model()
    )

    return loaded_index


if __name__ == "__main__":
    # 示例使用
    try:
        # 获取嵌入模型
        embed_model = embedding_service.get_embed_model()

        # 初始化管道
        pipeline = DataIngestionPipeline(embed_model=embed_model)

        # 示例书籍路径
        test_book_path = project_root / "readai/tests/data/test.epub"

        # 运行完整管道
        if test_book_path.exists():
            index = pipeline.run_pipeline(str(test_book_path))
            if index:
                print(f"数据摄入成功，索引对象: {index}")
            else:
                print("数据摄入失败，请查看日志了解详情")
        else:
            print(f"测试文件不存在: {test_book_path}")
    except Exception as e:
        logger.critical(f"运行示例时发生严重错误: {e!s}\n{traceback.format_exc()}")
        print(f"发生错误: {e!s}")

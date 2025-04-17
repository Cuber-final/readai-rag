# 该文件主要实现以下功能
# 1. 封装实现bm25+small2big的rqueryfusionRetriever，并支持指定几种重排序方法，RRF以及加权计算
# 2、结合llamaindex的query_engine，实现多轮问答，每轮都要使用检索器
import os
from enum import Enum
from pathlib import Path

import jieba
from dotenv import load_dotenv
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    RecursiveRetriever,
)
from llama_index.core.schema import NodeWithScore
from loguru import logger

from readai.components.retrievers import BM25Retriever
from readai.core.pipeline.ingestion import load_index

load_dotenv()

# 获取项目根目录路径
project_root = Path(os.getenv("PROJECT_ROOT"))
logger.add(f"{project_root}/logs/quering.log", rotation="10 MB")


class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""

    RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
    RELATIVE_SCORE = "relative_score"  # apply relative score fusion
    DIST_BASED_SCORE = "dist_based_score"  # apply distance-based score fusion
    SIMPLE = "simple"  # simple re-ordering of results based on original scores


def prepare_stop_words(stop_words_path: str = None) -> set:
    """准备停用词集合用于BM25检索器

    Args:
        stop_words_path: 停用词文件路径

    Returns:
        停用词集合
    """
    if not stop_words_path:
        stop_words_path = project_root / "readai/utils/baidu_stopwords.txt"

    custom_tokenizer = jieba.Tokenizer()
    custom_tokenizer.load_userdict(str(stop_words_path))

    # 加载stop_words为字符串列表
    try:
        with open(stop_words_path) as f:
            stop_words = f.readlines()
        stop_words = {word.strip() for word in stop_words if word.strip()}
        logger.info(f"加载停用词完成，共{len(stop_words)}个停用词")
        return stop_words
    except Exception as e:
        logger.error(f"加载停用词失败: {e!s}")
        return set()


class HybridFusionRetriever:
    """混合融合检索器，结合BM25和向量检索的优势"""

    def __init__(
        self,
        index: VectorStoreIndex,
        fusion_mode: FUSION_MODES = FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k: int = 5,
        num_queries: int = 1,
        use_rerank: bool = False,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        rerank_top_n: int = 5,
        verbose: bool = False,
        llm=None,
    ):
        """初始化混合融合检索器

        Args:
            index: 向量索引
            fusion_mode: 融合模式
            similarity_top_k: 每个检索器返回的结果数
            num_queries: 查询重写数量
            use_rerank: 是否使用重排序
            rerank_model: 重排序模型
            rerank_top_n: 重排序后返回的结果数
            verbose: 是否显示详细日志
            llm: 语言模型
        """
        self.index = index
        self.fusion_mode = fusion_mode
        self.similarity_top_k = similarity_top_k
        self.num_queries = num_queries
        self.use_rerank = use_rerank
        self.rerank_model = rerank_model
        self.rerank_top_n = rerank_top_n
        self.verbose = verbose
        self.llm = llm

        # 初始化分词器和停用词
        self.tokenizer = jieba.Tokenizer()
        stop_words_path = project_root / "readai/utils/baidu_stopwords.txt"
        self.stop_words = prepare_stop_words(stop_words_path)

        # 初始化检索器
        self._setup_retrievers()

        if self.verbose:
            logger.info(
                f"初始化混合融合检索器完成，融合模式: {fusion_mode}, 结果数: {similarity_top_k}"
            )

    def _setup_retrievers(self):
        """设置融合检索器的各个组件"""
        # 创建BM25检索器
        self.bm25_retriever = BM25Retriever.from_defaults(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
            tokenizer=self.tokenizer,
            stopwords=self.stop_words,
        )

        # 创建向量检索器
        vector_retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        # 创建small2big检索器
        self.small2big_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=self.index.docstore.docs,
            verbose=self.verbose,
        )

        # 创建融合检索器
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.small2big_retriever, self.bm25_retriever],
            mode=self.fusion_mode,
            verbose=self.verbose,
            similarity_top_k=self.similarity_top_k,
            num_queries=self.num_queries,
            llm=self.llm,
        )

        # 如果启用重排序，添加重排序处理器
        if self.use_rerank:
            try:
                self.reranker = SentenceTransformerRerank(
                    model=self.rerank_model, top_n=self.rerank_top_n
                )
            except Exception as e:
                logger.error(f"初始化重排序模型失败: {e!s}")
                self.use_rerank = False

    def retrieve(self, query_str: str) -> list[NodeWithScore]:
        """执行检索操作

        Args:
            query_str: 查询字符串

        Returns:
            检索结果列表
        """
        try:
            query_bundle = QueryBundle(query_str)

            # 使用融合检索器获取结果
            retrieval_results = self.fusion_retriever.retrieve(query_bundle)

            # 如果启用重排序，应用重排序
            if self.use_rerank and hasattr(self, "reranker"):
                nodes = self.reranker.postprocess_nodes(retrieval_results, query_bundle)
                if self.verbose:
                    logger.info(f"应用重排序后的结果数: {len(nodes)}")
                return nodes

            return retrieval_results

        except Exception as e:
            logger.error(f"检索过程中发生错误: {e!s}")
            return []

    def get_query_engine(self) -> RetrieverQueryEngine:
        """获取查询引擎

        Returns:
            基于当前检索器的查询引擎
        """
        response_synthesizer = CompactAndRefine.from_defaults(
            llm=self.llm,
            verbose=self.verbose,
        )

        return RetrieverQueryEngine.from_args(
            retriever=self.fusion_retriever,
            response_synthesizer=response_synthesizer,
            llm=self.llm,
        )


class MultiTurnConversation:
    """多轮对话管理器"""

    def __init__(
        self,
        collection_name: str,
        fusion_mode: FUSION_MODES = FUSION_MODES.RECIPROCAL_RANK,
        similarity_top_k: int = 5,
        use_rerank: bool = False,
        system_prompt: str = None,
        memory_size: int = 10,
        llm=None,
        verbose: bool = False,
    ):
        """初始化多轮对话管理器

        Args:
            collection_name: 集合名称，用于加载索引
            fusion_mode: 融合模式
            similarity_top_k: 检索结果数量
            use_rerank: 是否使用重排序
            system_prompt: 系统提示
            memory_size: 对话历史记忆大小
            llm: 语言模型
            verbose: 是否显示详细日志
        """
        self.collection_name = collection_name
        self.fusion_mode = fusion_mode
        self.similarity_top_k = similarity_top_k
        self.use_rerank = use_rerank
        self.system_prompt = (
            system_prompt or "你是一个有帮助的AI助手，将使用检索到的信息来回答问题。"
        )
        self.memory_size = memory_size
        self.llm = llm
        self.verbose = verbose

        # 初始化索引和检索器
        self._initialize()

    def _initialize(self):
        """初始化索引和检索组件"""
        try:
            # 加载索引
            self.index = load_index(self.collection_name)
            if not self.index:
                raise ValueError(f"无法加载索引: {self.collection_name}")

            # 初始化混合检索器
            self.retriever = HybridFusionRetriever(
                index=self.index,
                fusion_mode=self.fusion_mode,
                similarity_top_k=self.similarity_top_k,
                use_rerank=self.use_rerank,
                llm=self.llm,
                verbose=self.verbose,
            )

            # 初始化对话记忆
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=4096, memory_size=self.memory_size
            )

            # 初始化聊天引擎
            self.chat_engine = ContextChatEngine.from_defaults(
                retriever=self.retriever.fusion_retriever,
                llm=self.llm,
                memory=self.memory,
                system_prompt=self.system_prompt,
                verbose=self.verbose,
            )

            logger.info(f"初始化多轮对话管理器完成，集合: {self.collection_name}")

        except Exception as e:
            logger.error(f"初始化多轮对话管理器失败: {e!s}")
            raise

    def chat(self, query: str) -> str:
        """进行对话

        Args:
            query: 用户查询

        Returns:
            助手响应
        """
        try:
            response = self.chat_engine.chat(query)
            return response.response
        except Exception as e:
            logger.error(f"对话过程中发生错误: {e!s}")
            return f"抱歉，处理您的查询时发生错误: {e!s}"

    def reset(self):
        """重置对话上下文"""
        try:
            self.memory.reset()
            logger.info("对话上下文已重置")
        except Exception as e:
            logger.error(f"重置对话上下文失败: {e!s}")

    def get_chat_history(self) -> list[ChatMessage]:
        """获取对话历史

        Returns:
            对话消息列表
        """
        return self.memory.get_messages()

    def add_message(self, role: MessageRole, content: str):
        """添加消息到对话历史

        Args:
            role: 消息角色
            content: 消息内容
        """
        try:
            self.memory.put(ChatMessage(role=role, content=content))
        except Exception as e:
            logger.error(f"添加消息到对话历史失败: {e!s}")


if __name__ == "__main__":
    from llama_index.llms.deepseek import DeepSeek

    from readai.components.embedding import embedding_service

    # 加载环境变量
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model_name = os.getenv("DEEPSEEK_MODEL")

    # 初始化模型
    embed_model = embedding_service.get_embed_model()
    llm = DeepSeek(model=model_name, api_key=api_key)

    # 示例使用
    try:
        # 初始化多轮对话
        conversation = MultiTurnConversation(
            collection_name="hybrid_nodes",
            fusion_mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=5,
            use_rerank=True,
            llm=llm,
            verbose=True,
        )

        # 模拟对话
        queries = [
            "非暴力沟通有哪些核心观点？",
            "请详细解释一下非暴力沟通的第一个核心观点",
            "你能举个例子说明这个观点在日常生活中的应用吗？",
        ]

        for query in queries:
            print(f"\n用户: {query}")
            response = conversation.chat(query)
            print(f"助手: {response}")

    except Exception as e:
        print(f"示例运行失败: {e!s}")

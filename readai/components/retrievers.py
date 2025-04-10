from collections.abc import Callable
from typing import Any, cast

import bm25s
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from rank_bm25 import BM25Okapi

from readai.core.pipeline.ingestion import get_node_content


class SentenceWindowRetriever(BaseRetriever):
    """句子窗口检索器

    专门用于检索与选中句子相关的上下文，保持句子之间的上下文关系
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        window_size: int = 3,
        similarity_top_k: int = 5,
        rerank_top_n: int = 3,
        reranker_model: str = "BAAI/bge-reranker-base",
        window_metadata_key: str = "window",
        original_text_metadata_key: str = "original_text",
    ):
        """初始化句子窗口检索器

        Args:
            index: 向量索引
            window_size: 句子窗口大小
            similarity_top_k: 向量检索返回的最大文档数
            rerank_top_n: 重排序后保留的文档数
            reranker_model: 重排序模型名称
            window_metadata_key: 窗口信息存储的元数据键
            original_text_metadata_key: 原始文本存储的元数据键
        """
        self.index = index
        self.window_size = window_size
        self.similarity_top_k = similarity_top_k
        self.window_metadata_key = window_metadata_key
        self.original_text_metadata_key = original_text_metadata_key

        # 初始化检索器和重排序器
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

        self.reranker = None
        if rerank_top_n > 0:
            self.reranker = SentenceTransformerRerank(
                top_n=rerank_top_n, model=reranker_model
            )

        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """异步检索与查询相关的句子及其上下文

        Args:
            query_bundle: 查询包，可能包含选中的句子信息

        Returns:
            相关句子节点列表及其分数
        """
        # 获取查询信息
        query_str = query_bundle.query_str

        # 检查是否有选中的句子
        selected_sentence = (
            query_bundle.custom_embedding_strs[0]
            if query_bundle.custom_embedding_strs
            else None
        )

        # 构建查询文本
        if selected_sentence:
            # 如果有选中的句子，将其作为上下文加入查询
            enhanced_query = f"{query_str} 上下文: {selected_sentence}"
            query_bundle = QueryBundle(query_str=enhanced_query)

        # 执行向量检索
        nodes = await self.retriever.aretrieve(query_bundle)

        # 如果有选中的句子，确保包含选中的句子所在窗口
        if selected_sentence:
            # 从结果中找到匹配的节点
            selected_node = self._find_selected_sentence_node(selected_sentence, nodes)

            # 如果未找到匹配节点但存在选中句子，则创建新查询
            if not selected_node:
                # 直接用选中句子进行检索
                direct_query = QueryBundle(query_str=selected_sentence)
                selected_nodes = await self.retriever.aretrieve(direct_query)

                # 将选中句子的相关节点添加到结果中
                if selected_nodes:
                    # 确保不重复添加
                    existing_ids = {node.node.node_id for node in nodes}
                    for node in selected_nodes:
                        if node.node.node_id not in existing_ids:
                            nodes.append(node)

        # 应用重排序
        if self.reranker and nodes:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

        # 扩充上下文窗口
        nodes = self._expand_context_windows(nodes)

        return nodes

    def _find_selected_sentence_node(
        self, selected_sentence: str, nodes: list[NodeWithScore]
    ) -> NodeWithScore | None:
        """查找包含选中句子的节点

        Args:
            selected_sentence: 选中的句子
            nodes: 检索到的节点列表

        Returns:
            包含选中句子的节点，如果未找到则返回None
        """
        for node in nodes:
            # 检查节点内容或元数据中是否包含选中句子
            if selected_sentence in node.node.get_content():
                return node

            # 检查窗口元数据
            window_text = node.node.metadata.get(self.window_metadata_key, "")
            if selected_sentence in window_text:
                return node

        return None

    def _expand_context_windows(
        self, nodes: list[NodeWithScore]
    ) -> list[NodeWithScore]:
        """扩展节点的上下文窗口

        为每个节点添加父节点和相邻节点的信息

        Args:
            nodes: 原始检索节点列表

        Returns:
            扩展了上下文的节点列表
        """
        # 这里可以实现更复杂的上下文扩展逻辑
        # 如果节点已经包含窗口信息，则可以直接使用
        return nodes

    def get_sentence_context(self, selected_sentence: str) -> dict[str, Any]:
        """获取选中句子的详细上下文信息

        Args:
            selected_sentence: 用户选中的句子

        Returns:
            包含句子上下文的字典
        """
        # 创建查询
        query_bundle = QueryBundle(query_str=selected_sentence)

        # 同步执行检索
        nodes = self.retrieve(query_bundle)

        # 从结果中提取上下文信息
        context_info = {
            "sentence": selected_sentence,
            "context_before": [],
            "context_after": [],
            "related_info": [],
        }

        # 提取上下文
        for node in nodes:
            content = node.node.get_content()
            window = node.node.metadata.get(self.window_metadata_key, "")

            # 将窗口信息解析为上下文
            if selected_sentence in content:
                # 尝试分割窗口获取前后文
                try:
                    parts = window.split(selected_sentence)
                    if len(parts) > 1:
                        context_info["context_before"].append(parts[0].strip())
                        context_info["context_after"].append(parts[1].strip())
                    else:
                        context_info["related_info"].append(window)
                except:
                    context_info["related_info"].append(window)

        return context_info

    def retrieve_with_selected_sentence(
        self, query: str, selected_sentence: str
    ) -> list[NodeWithScore]:
        """使用选中的句子和查询进行联合检索

        Args:
            query: 用户的查询
            selected_sentence: 用户选中的句子

        Returns:
            检索结果
        """
        # 创建带有选中句子的查询包
        query_bundle = QueryBundle(
            query_str=query, custom_embedding_strs=[selected_sentence]
        )

        # 执行检索
        return self.retrieve(query_bundle)


class EnhancedSentenceRetriever(BaseRetriever):
    """增强的句子检索器，支持多粒度检索"""

    def __init__(
        self,
        vector_store,
        sentence_embed_model,
        paragraph_embed_model=None,
        similarity_top_k: int = 3,
        include_paragraph_context: bool = True,
    ):
        self.vector_store = vector_store
        self.sentence_embed_model = sentence_embed_model
        self.paragraph_embed_model = paragraph_embed_model or sentence_embed_model
        self.similarity_top_k = similarity_top_k
        self.include_paragraph_context = include_paragraph_context

        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # 获取查询信息
        query_str = query_bundle.query_str
        selected_sentence = (
            query_bundle.custom_embedding_strs[0]
            if query_bundle.custom_embedding_strs
            else None
        )

        # 检索相关句子
        sentence_nodes = await self._retrieve_sentences(query_str, selected_sentence)

        # 如果需要段落上下文，获取相关段落
        if self.include_paragraph_context:
            paragraph_nodes = await self._retrieve_paragraphs(
                sentence_nodes, query_str, selected_sentence
            )

            # 合并结果并排序
            all_nodes = self._merge_results(sentence_nodes, paragraph_nodes)
            return all_nodes

        return sentence_nodes

    async def _retrieve_sentences(
        self, query: str, selected_sentence: str | None = None
    ) -> list[NodeWithScore]:
        """检索相关句子"""
        # 构建检索查询
        query_text = f"{query} {selected_sentence}" if selected_sentence else query
        query_embedding = self.sentence_embed_model.get_query_embedding(query_text)

        # 执行检索
        # 这里假设vector_store已经包含了句子级别的节点
        # 实际实现需要根据您的存储结构调整
        results = await self.vector_store.asimilarity_search_with_score(
            query_embedding,
            k=self.similarity_top_k,
            filter={"node_type": "sentence"},  # 假设有这样的过滤条件
        )

        return results

    async def _retrieve_paragraphs(
        self,
        sentence_nodes: list[NodeWithScore],
        query: str,
        selected_sentence: str | None = None,
    ) -> list[NodeWithScore]:
        """检索相关段落"""
        # 从句子节点获取段落ID
        paragraph_ids = set()
        for node in sentence_nodes:
            p_id = node.node.metadata.get("paragraph_id")
            if p_id:
                paragraph_ids.add(p_id)

        # 直接获取这些段落
        paragraph_nodes = []
        for p_id in paragraph_ids:
            # 这里需要根据您的存储结构获取段落节点
            # 以下是示例逻辑
            p_node = await self.vector_store.get_node(p_id)
            if p_node:
                # 分数可以基于相关句子的分数
                related_sentence_scores = [
                    n.score
                    for n in sentence_nodes
                    if n.node.metadata.get("paragraph_id") == p_id
                ]
                avg_score = (
                    sum(related_sentence_scores) / len(related_sentence_scores)
                    if related_sentence_scores
                    else 0.5
                )
                paragraph_nodes.append(NodeWithScore(node=p_node, score=avg_score))

        return paragraph_nodes

    def _merge_results(
        self, sentence_nodes: list[NodeWithScore], paragraph_nodes: list[NodeWithScore]
    ) -> list[NodeWithScore]:
        """合并句子和段落结果"""
        # 确保不重复
        seen_ids = set()
        merged = []

        # 先添加句子节点
        for node in sentence_nodes:
            merged.append(node)
            seen_ids.add(node.node.node_id)

        # 再添加段落节点
        for node in paragraph_nodes:
            if node.node.node_id not in seen_ids:
                # 段落节点的分数可以乘以因子使其略低于句子节点
                node.score *= 0.9
                merged.append(node)
                seen_ids.add(node.node.node_id)

        # 按分数排序
        return sorted(merged, key=lambda x: x.score, reverse=True)


class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 2,
        filters=None,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self.filters = filters
        super().__init__()

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding,
            similarity_top_k=self._similarity_top_k,
            # filters=self.filters, # qdrant 使用llama_index filter会有问题，原因未知
        )
        query_result = await self._vector_store.aquery(
            vector_store_query,
            qdrant_filters=self.filters,  # 需要查找qdrant相关用法
        )

        node_with_scores = []
        for node, similarity in zip(
            query_result.nodes, query_result.similarities, strict=False
        ):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding,
            similarity_top_k=self._similarity_top_k,
        )
        query_result = self._vector_store.query(
            vector_store_query,
            qdrant_filters=self.filters,  # 需要查找qdrant相关用法
        )

        node_with_scores = []
        for node, similarity in zip(
            query_result.nodes, query_result.similarities, strict=False
        ):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


def tokenize_and_remove_stopwords(tokenizer, text, stopwords):
    words = tokenizer.cut(text)
    filtered_words = [word for word in words if word not in stopwords and word != " "]
    return filtered_words


# using jieba to split sentence and remove meaningless words
class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[BaseNode],
        tokenizer: Callable[[str], list[str]],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: CallbackManager | None = None,
        objects: list[IndexNode] | None = None,
        object_map: dict | None = None,
        verbose: bool = False,
        stopwords: list[str] = [""],
        embed_type: int = 0,
        bm25_type: int = 0,
    ) -> None:
        self._nodes = nodes
        self._tokenizer = tokenizer
        self._similarity_top_k = similarity_top_k
        self.embed_type = embed_type
        self._corpus = [
            tokenize_and_remove_stopwords(
                self._tokenizer,
                get_node_content(node, embed_type=0),
                stopwords=stopwords,
            )
            for node in self._nodes
        ]
        # self._corpus = [self._tokenizer(node.get_content()) for node in self._nodes]
        self.bm25_type = bm25_type
        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25
        if self.bm25_type == 1:
            self.bm25 = bm25s.BM25(
                k1=self.k1,
                b=self.b,
            )
            self.bm25.index(self._corpus)
        else:
            self.bm25 = BM25Okapi(
                self._corpus,
                k1=self.k1,
                b=self.b,
                epsilon=self.epsilon,
            )
        self.filter_dict = None
        self.stopwords = stopwords
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def get_scores(self, query, docs=None):
        if docs is None:
            bm25 = self.bm25
        else:
            corpus = [
                tokenize_and_remove_stopwords(
                    self._tokenizer, doc, stopwords=self.stopwords
                )
                for doc in docs
            ]
            if self.bm25_type == 1:
                bm25 = bm25s.BM25(
                    k1=self.k1,
                    b=self.b,
                )
                bm25.index(corpus)
            else:
                bm25 = BM25Okapi(
                    corpus,
                    k1=self.k1,
                    b=self.b,
                    epsilon=self.epsilon,
                )
        tokenized_query = tokenize_and_remove_stopwords(
            self._tokenizer, query, stopwords=self.stopwords
        )
        scores = bm25.get_scores(tokenized_query)
        return scores

    @classmethod
    def from_defaults(
        cls,
        index: VectorStoreIndex | None = None,
        nodes: list[BaseNode] | None = None,
        docstore: BaseDocumentStore | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        stopwords: list[str] = [""],
        embed_type: int = 0,
        bm25_type: int = 0,  # 0-->official bm25-Okapi 1-->bm25s
    ) -> "BM25Retriever":
        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast("list[BaseNode]", list(docstore.docs.values()))

        assert nodes is not None, (
            "Please pass exactly one of index, nodes, or docstore."
        )

        tokenizer = tokenizer
        return cls(
            nodes=nodes,
            tokenizer=tokenizer,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            stopwords=stopwords,
            embed_type=embed_type,
            bm25_type=bm25_type,
        )

    def filter(self, scores):
        top_n = scores.argsort()[::-1]
        nodes: list[NodeWithScore] = []
        for ix in top_n:
            if scores[ix] <= 0:
                break
            flag = True
            if self.filter_dict is not None:
                for key, value in self.filter_dict.items():
                    if self._nodes[ix].metadata[key] != value:
                        flag = False
                        break
            if flag:
                nodes.append(
                    NodeWithScore(node=self._nodes[ix], score=float(scores[ix]))
                )
            if len(nodes) == self._similarity_top_k:
                break

        # add nodes sort in BM25Retriever
        nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if query_bundle.custom_embedding_strs or query_bundle.embedding:
            logger.warning("BM25Retriever does not support embeddings, skipping...")

        query = query_bundle.query_str
        scores = self.get_scores(query)
        nodes = self.filter(scores)

        return nodes


# 自己封装一个rerank retriever
class RerankRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: list[BaseRetriever],
        top_n: int = 3,
        rerank_model: str = "BAAI/bge-reranker-base",
    ):
        self.retrievers = retrievers
        self.top_n = top_n
        self.rerank_model = rerank_model

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        all_nodes = []
        for retriever in self.retrievers:
            nodes = await retriever.aretrieve(query_bundle)
            # 使用rerank模型对所有节点进行重排序
            reranked_nodes = self.rerank_model.rerank(nodes, query_bundle.query_str)
            all_nodes.extend(reranked_nodes)

        return reranked_nodes[: self.top_n]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        all_nodes = []
        for retriever in self.retrievers:
            nodes = retriever.retrieve(query_bundle)
            all_nodes.extend(nodes)
        return all_nodes[: self.top_n]

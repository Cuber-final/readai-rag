import asyncio
import math
from typing import Dict, List, Optional, Tuple

from FlagEmbedding import FlagReranker
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle


class MyReRankRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        model_name_or_path: str = "bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        devices: str
        | list[str]
        | list[int]
        | None = None,  # specify devices, such as ["cuda:0"] or ["0"],
    ) -> None:
        self.similarity_top_k = similarity_top_k
        self.use_async = use_async
        self._retrievers = retrievers
        self.devices = devices
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.reranker = FlagReranker(
            model_name_or_path=model_name_or_path,
            use_fp16=use_fp16,
            devices=devices,
            batch_size=batch_size,
            normalize=True,
        )
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def _rerank_results(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_score in results.values():
            for node_with_score in nodes_with_score:
                node_hash = node_with_score.node.hash
                if node_hash not in all_nodes:
                    all_nodes[node_hash] = node_with_score

        # 获取查询文本（所有results的key都共享同一个query）
        query_text = next(iter(results.keys()))[0] if results else ""

        # 准备reranker输入 (query, node_text) 对
        sentence_pairs = [
            (query_text, node_with_score.text) for node_with_score in all_nodes.values()
        ]

        rerank_scores = self.reranker.compute_score(sentence_pairs)

        reranked_nodes: list[NodeWithScore] = []
        for node_with_score, rerank_score in zip(all_nodes.values(), rerank_scores):
            new_score = rerank_score if not math.isnan(rerank_score) else 0.0
            reranked_nodes.append(
                NodeWithScore(
                    node=node_with_score.node,
                    score=new_score,
                )
            )

        return sorted(reranked_nodes, key=lambda x: x.score, reverse=True)

    def _run_nested_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = run_async_tasks(tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    async def _run_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = await asyncio.gather(*tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    def _run_sync_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                results[(query.query_str, i)] = retriever.retrieve(query)

        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        return self._rerank_results(results)[: self.similarity_top_k]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]

        results = await self._run_async_queries(queries)

        return self._rerank_results(results)[: self.similarity_top_k]

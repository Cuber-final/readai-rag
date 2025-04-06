#!/usr/bin/env python

"""上下文检索对比实验

该脚本实现了基于上下文增强的检索方法，并对比了不同检索策略的效果:
1. 基础向量检索
2. BM25检索
3. 向量+BM25混合检索
4. 上下文增强的向量检索
5. 上下文增强的BM25检索
6. 上下文增强的混合检索

通过对不同检索方法的对比，展示上下文增强对检索质量的提升效果。
"""

import copy
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# LLamaIndex 相关导入
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    generate_question_context_pairs,
    get_retrieval_results_df,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from tqdm import tqdm

from readai.components.epub2md_loader import preprocess_book

# 本地组件导入
from readai.components.prompts import QA_GENERATE_PROMPT_TMPL_ZH

# 加载环境变量
load_dotenv()

# 日志配置
log_path = Path("logs/contextual_retrieval.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, level="INFO", encoding="utf-8")

# 项目路径
project_root = Path(os.getenv("PROJECT_ROOT"))
data_path = project_root / "readai/tests/data"


class ContextualRetrieverExperiment:
    """上下文检索实验类，用于比较不同检索方法的效果"""

    def __init__(
        self,
        book_path=None,
        llm_model="deepseek-chat",
        embed_model_name="quentinz/bge-large-zh-v1.5",
        similarity_top_k=5,
    ):
        """初始化实验环境

        Args:
            book_path: 书籍路径
            llm_model: 大语言模型名称
            embed_model_name: 嵌入模型名称
            similarity_top_k: 检索结果数量
        """
        self.book_path = book_path
        self.similarity_top_k = similarity_top_k

        # 设置LLM模型
        self.llm = DeepSeek(
            model=llm_model,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            api_base="https://api.deepseek.com",
        )

        # 设置嵌入模型
        self.embed_model = OllamaEmbedding(
            model_name=embed_model_name, base_url="http://localhost:11434"
        )
        Settings.embed_model = self.embed_model

        # 设置Qdrant客户端
        from qdrant_client import AsyncQdrantClient, QdrantClient

        self.client = QdrantClient(url="http://localhost:6333")
        self.aclient = AsyncQdrantClient(url="http://localhost:6333")

        # 设置重排序模型
        self.reranker = CohereRerank(
            api_key=os.environ.get("COHERE_API_KEY", ""), top_n=self.similarity_top_k
        )

        # 文档和节点
        self.documents = None
        self.nodes = None
        self.contextual_nodes = None

        # 索引和检索器
        self.base_index = None
        self.base_retriever = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        self.contextual_index = None
        self.contextual_retriever = None
        self.contextual_bm25_retriever = None
        self.contextual_hybrid_retriever = None

        # 评估数据
        self.eval_dataset = None

    def load_data(self):
        """加载文档数据"""
        if self.book_path is None:
            # 默认使用测试数据
            self.book_path = data_path / "renzhi.epub"

        logger.info(f"加载文档: {self.book_path}")

        metadata, cleaned_path = preprocess_book(str(self.book_path))
        loader = UnstructuredMarkdownLoader(cleaned_path, mode="elements")
        raw_docs = loader.load()
        # 处理为llama_index documents
        self.documents = [
            Document(
                text=doc.page_content,
                metadata={
                    "category": doc.metadata.get("category", ""),
                    "filename": doc.metadata.get("filename", ""),
                },
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            for doc in raw_docs
        ]

        logger.info(f"加载了 {len(self.documents)} 个文档")
        return self.documents

    def create_nodes(self, chunk_size=1024, chunk_overlap=50):
        """创建文本节点"""
        # 创建节点解析器
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        # 解析文档成节点
        self.nodes = node_parser.get_nodes_from_documents(self.documents)

        # 设置节点ID (确保一致性)
        for idx, node in enumerate(self.nodes):
            node.id_ = f"node_{idx}"

        logger.info(f"创建了 {len(self.nodes)} 个基础节点")
        return self.nodes

    def create_contextual_nodes(self):
        """创建带上下文的增强节点"""
        if self.nodes is None:
            raise ValueError("请先调用create_nodes()创建基础节点")

        # 从第一个文档中提取全部文本作为完整上下文
        whole_document = "\n\n".join([doc.text for doc in self.documents])

        # 准备提示模板
        prompt_document = """<document>
        {WHOLE_DOCUMENT}
        </document>"""

        prompt_chunk = """这是我们需要理解的文本片段:
        <chunk>
        {CHUNK_CONTENT}
        </chunk>
        请提供一个简短精确的描述，帮助我们理解这个片段在整个文档中的位置和上下文。只需提供简洁的描述即可。"""

        # 复制节点并添加上下文信息
        self.contextual_nodes = []

        for node in tqdm(self.nodes, desc="创建上下文节点"):
            new_node = copy.deepcopy(node)

            # 使用LLM生成上下文描述
            messages = [
                {"role": "system", "content": "你是一位擅长总结和提取文本重点的助手。"},
                {
                    "role": "user",
                    "content": prompt_document.format(WHOLE_DOCUMENT=whole_document)
                    + "\n"
                    + prompt_chunk.format(CHUNK_CONTENT=node.text),
                },
            ]

            context = self.llm.chat(messages).message.content
            new_node.metadata["context"] = context
            self.contextual_nodes.append(new_node)

        logger.info(f"创建了 {len(self.contextual_nodes)} 个上下文增强节点")
        return self.contextual_nodes

    def create_indices_and_retrievers(self):
        """创建各种索引和检索器"""
        # 1. 基础向量索引和检索器
        vector_store_base = QdrantVectorStore(
            client=self.client,
            aclient=self.aclient,
            collection_name="base_nodes",
        )

        collection_name = "base_nodes"
        if self.client.collection_exists(collection_name):
            logger.info(f"集合 '{collection_name}' 已存在，加载现有索引")
            self.base_index = VectorStoreIndex.from_vector_store(
                vector_store_base, embed_model=self.embed_model
            )
        else:
            logger.info(f"集合 '{collection_name}' 不存在，创建新索引")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store_base
            )
            self.base_index = VectorStoreIndex(
                self.nodes,
                embed_model=self.embed_model,
                storage_context=storage_context,
                show_progress=True,
            )

        self.base_retriever = self.base_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

        # 2. BM25检索器
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=self.similarity_top_k,
        )

        # 3. 混合检索器 (向量+BM25+重排序)
        self.hybrid_retriever = HybridRetriever(
            self.base_retriever,
            self.bm25_retriever,
            self.reranker,
        )

        # 4. 上下文增强向量索引和检索器
        if self.contextual_nodes:
            vector_store_contextual = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name="contextual_nodes",
            )

            collection_name = "contextual_nodes"
            if self.client.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在，加载现有索引")
                self.contextual_index = VectorStoreIndex.from_vector_store(
                    vector_store_contextual, embed_model=self.embed_model
                )
            else:
                logger.info(f"集合 '{collection_name}' 不存在，创建新索引")
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store_contextual
                )
                self.contextual_index = VectorStoreIndex(
                    self.contextual_nodes,
                    embed_model=self.embed_model,
                    storage_context=storage_context,
                    show_progress=True,
                )

            self.contextual_retriever = self.contextual_index.as_retriever(
                similarity_top_k=self.similarity_top_k
            )

            # 5. 上下文增强BM25检索器
            self.contextual_bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.contextual_nodes,
                similarity_top_k=self.similarity_top_k,
            )

            # 6. 上下文增强混合检索器
            self.contextual_hybrid_retriever = HybridRetriever(
                self.contextual_retriever,
                self.contextual_bm25_retriever,
                self.reranker,
            )

        logger.info("所有索引和检索器创建完成")

    def create_evaluation_dataset(self, num_questions=50):
        """创建评估数据集"""
        if self.nodes is None:
            raise ValueError("请先调用create_nodes()创建基础节点")

        # 选择一部分节点用于生成问题
        eval_nodes = (
            self.nodes[:num_questions]
            if len(self.nodes) > num_questions
            else self.nodes
        )

        logger.info(f"使用 {len(eval_nodes)} 个节点创建评估数据集")
        self.eval_dataset = generate_question_context_pairs(
            eval_nodes, llm=self.llm, qa_generate_prompt_tmpl=QA_GENERATE_PROMPT_TMPL_ZH
        )

        # 保存评估数据集
        output_path = "evaluation_dataset.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_dataset.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"评估数据集已保存到 {output_path}")
        return self.eval_dataset

    async def evaluate_retrievers(self):
        """评估各种检索器的性能"""
        if self.eval_dataset is None:
            raise ValueError("请先调用create_evaluation_dataset()创建评估数据集")

        results = {}
        names = []
        results_list = []

        # 准备评估的检索器列表
        retrievers = {
            "向量检索": self.base_retriever,
            "BM25检索": self.bm25_retriever,
            "混合检索": self.hybrid_retriever,
        }

        # 如果有上下文增强检索器，也加入评估
        if self.contextual_retriever:
            retrievers.update(
                {
                    "上下文增强向量检索": self.contextual_retriever,
                    "上下文增强BM25检索": self.contextual_bm25_retriever,
                    "上下文增强混合检索": self.contextual_hybrid_retriever,
                }
            )

        # 评估每个检索器
        for name, retriever in retrievers.items():
            logger.info(f"评估 {name}")
            evaluator = RetrieverEvaluator.from_metric_names(
                ["hit_rate", "mrr"], retriever=retriever
            )

            # 异步评估
            eval_results = await evaluator.aevaluate_dataset(
                self.eval_dataset, show_progress=True
            )

            names.append(name)
            results_list.append(eval_results)

            # 计算并保存基本指标
            metrics = {}
            metric_dicts = [result.metric_vals_dict for result in eval_results]
            metrics_df = pd.DataFrame(metric_dicts)

            for metric in ["hit_rate", "mrr"]:
                metrics[metric] = metrics_df[metric].mean()

            results[name] = metrics
            logger.info(
                f"{name} - 命中率: {metrics['hit_rate']:.4f}, MRR: {metrics['mrr']:.4f}"
            )

        # 生成对比表格
        results_df = get_retrieval_results_df(names, results_list)

        # 保存结果
        results_df.to_csv("retriever_evaluation_results.csv", index=False)
        logger.info("评估结果已保存到 retriever_evaluation_results.csv")

        return results_df

    def test_retrieval(self, query, display_results=True):
        """测试各种检索器在特定查询上的表现"""
        results = {}

        # 准备测试的检索器
        retrievers = {
            "向量检索": self.base_retriever,
            "BM25检索": self.bm25_retriever,
            "混合检索": self.hybrid_retriever,
        }

        # 如果有上下文增强检索器，也加入测试
        if self.contextual_retriever:
            retrievers.update(
                {
                    "上下文增强向量检索": self.contextual_retriever,
                    "上下文增强BM25检索": self.contextual_bm25_retriever,
                    "上下文增强混合检索": self.contextual_hybrid_retriever,
                }
            )

        # 测试每个检索器
        for name, retriever in retrievers.items():
            print(f"\n=== {name} ===")
            start_time = time.time()
            retrieved_nodes = retriever.retrieve(query)
            elapsed = time.time() - start_time

            print(f"检索时间: {elapsed:.4f}秒, 检索到 {len(retrieved_nodes)} 个节点")

            # 显示前3个结果
            if display_results:
                for i, node in enumerate(retrieved_nodes[:3]):
                    print(
                        f"\n结果 #{i + 1} (相似度: {getattr(node, 'score', 'N/A'):.4f})"
                    )
                    print("-" * 50)
                    print(
                        node.text[:300] + "..." if len(node.text) > 300 else node.text
                    )
                    print("-" * 50)

            results[name] = retrieved_nodes

        return results


class HybridRetriever(BaseRetriever):
    """混合检索器，结合向量检索、BM25检索和重排序"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: CohereRerank,
    ) -> None:
        """初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            reranker: 重排序器
        """
        self._vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker

        super().__init__()

    def _retrieve(self, query_bundle):
        """检索节点"""
        # 从向量索引检索
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        # 从BM25检索
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)

        # 合并结果
        all_nodes = []
        all_nodes.extend(vector_nodes)
        all_nodes.extend(bm25_nodes)

        # 使用重排序
        reranked_nodes = self.reranker.postprocess_nodes(all_nodes, query_bundle)

        return reranked_nodes


def main():
    experiment = ContextualRetrieverExperiment()
    experiment.load_data()
    experiment.create_nodes()
    experiment.create_contextual_nodes()
    experiment.create_indices_and_retrievers()

    test_query = "非暴力沟通的四个要素是什么？"
    experiment.test_retrieval(test_query)

    # experiment.create_evaluation_dataset()
    # 评估检索器性能（异步操作）
    # import asyncio
    # results_df = asyncio.run(experiment.evaluate_retrievers())
    # print(results_df)


if __name__ == "__main__":
    main()

from collections.abc import Sequence
from typing import Any

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import OpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser

# from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode


class MultiLevelNodeParser(IngestionPipeline):
    """多层级节点解析器，符合LlamaIndex设计范式

    作为IngestionPipeline的子类，可以直接集成到处理流程中
    """

    def __init__(
        self,
        sentence_window_size: int = 3,
        paragraph_sep: str = "\n\n",
        generate_summaries: bool = True,
        llm=None,
    ):
        # 段落级解析器
        paragraph_parser = SentenceWindowNodeParser.from_defaults(
            window_size=0,
            separator=paragraph_sep,
        )

        # 句子级解析器
        sentence_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        # 构建转换链
        transformations = [paragraph_parser, sentence_parser]

        # 如果需要生成摘要，添加摘要提取器
        if generate_summaries:
            llm = llm or OpenAI()
            summary_extractor = SummaryExtractor(
                llm=llm,
                summaries=["paragraph_summary"],
            )
            transformations.append(summary_extractor)

        # 添加关系构建器
        transformations.append(RelationshipBuilder())

        # 初始化管道
        super().__init__(transformations=transformations)


class RelationshipBuilder:
    """节点关系构建器

    建立句子和段落之间的父子关系
    """

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """执行关系构建"""
        # 区分段落节点和句子节点
        paragraph_nodes = []
        sentence_nodes = []

        for node in nodes:
            # 通过节点大小或元数据判断节点类型
            if len(node.text) > 200 or "separator" in node.metadata:  # 假设段落较长
                node.metadata["node_type"] = "paragraph"
                paragraph_nodes.append(node)
            else:
                node.metadata["node_type"] = "sentence"
                sentence_nodes.append(node)

        # 构建关系
        self._build_relationships(sentence_nodes, paragraph_nodes)

        return nodes

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """异步执行关系构建"""
        return self.__call__(nodes, **kwargs)

    def _build_relationships(
        self, sentence_nodes: list[TextNode], paragraph_nodes: list[TextNode]
    ):
        """建立句子与其所属段落的关系"""
        # 为段落节点创建索引
        paragraph_index = {}
        for p_node in paragraph_nodes:
            paragraph_index[p_node.text] = p_node

        # 为每个句子找到所属段落
        for s_node in sentence_nodes:
            sentence_text = s_node.text
            for p_text, p_node in paragraph_index.items():
                if sentence_text in p_text:
                    # 建立双向关系
                    s_node.relationships[NodeRelationship.PARENT] = p_node.node_id

                    # 记录段落ID到句子元数据
                    s_node.metadata["paragraph_id"] = p_node.node_id

                    # 如果段落有摘要，添加到句子元数据
                    if "summary" in p_node.metadata:
                        s_node.metadata["paragraph_summary"] = p_node.metadata[
                            "summary"
                        ]
                    break


def _add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list = parent_node.relationships.get(NodeRelationship.CHILD, [])
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[NodeRelationship.PARENT] = (
        parent_node.as_related_node_info()
    )


def get_leaf_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """Get leaf nodes."""
    leaf_nodes = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            leaf_nodes.append(node)
    return leaf_nodes


def get_root_nodes(nodes: list[BaseNode]) -> list[BaseNode]:
    """Get root nodes."""
    root_nodes = []
    for node in nodes:
        if NodeRelationship.PARENT not in node.relationships:
            root_nodes.append(node)
    return root_nodes


def get_child_nodes(nodes: list[BaseNode], all_nodes: list[BaseNode]) -> list[BaseNode]:
    """Get child nodes of nodes from given all_nodes."""
    children_ids = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            continue

        children_ids.extend(
            [r.node_id for r in node.relationships[NodeRelationship.CHILD]]
        )

    child_nodes = []
    for candidate_node in all_nodes:
        if candidate_node.node_id not in children_ids:
            continue
        child_nodes.append(candidate_node)

    return child_nodes


def get_deeper_nodes(nodes: list[BaseNode], depth: int = 1) -> list[BaseNode]:
    """Get children of root nodes in given nodes that have given depth."""
    if depth < 0:
        raise ValueError("Depth cannot be a negative number!")
    root_nodes = get_root_nodes(nodes)
    if not root_nodes:
        raise ValueError("There is no root nodes in given nodes!")

    deeper_nodes = root_nodes
    for _ in range(depth):
        deeper_nodes = get_child_nodes(deeper_nodes, nodes)

    return deeper_nodes


class HierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).

    For instance, this may return a list of nodes like:
    - list of top-level nodes with chunk size 2048
    - list of second-level nodes, where each node is a child of a top-level node,
      chunk size 512
    - list of third-level nodes, where each node is a child of a second-level node,
      chunk size 128
    """

    chunk_sizes: list[int] | None = Field(
        default=None,
        description=(
            "The chunk sizes to use when splitting documents, in order of level."
        ),
    )
    node_parser_ids: list[str] = Field(
        default_factory=list,
        description=(
            "List of ids for the node parsers to use when splitting documents, "
            + "in order of level (first id used for first level, etc.)."
        ),
    )
    node_parser_map: dict[str, NodeParser] = Field(
        description="Map of node parser id to node parser.",
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_sizes: list[int] | None = None,
        chunk_overlap: int = 20,
        node_parser_ids: list[str] | None = None,
        node_parser_map: dict[str, NodeParser] | None = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
    ) -> "HierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        if node_parser_ids is None:
            if chunk_sizes is None:
                chunk_sizes = [2048, 512, 128]

            node_parser_ids = [f"chunk_size_{chunk_size}" for chunk_size in chunk_sizes]
            node_parser_map = {}
            for chunk_size, node_parser_id in zip(
                chunk_sizes, node_parser_ids, strict=False
            ):
                node_parser_map[node_parser_id] = SentenceSplitter(
                    chunk_size=chunk_size,
                    callback_manager=callback_manager,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    include_prev_next_rel=include_prev_next_rel,
                )
        else:
            if chunk_sizes is not None:
                raise ValueError("Cannot specify both node_parser_ids and chunk_sizes.")
            if node_parser_map is None:
                raise ValueError(
                    "Must specify node_parser_map if using node_parser_ids."
                )

        return cls(
            chunk_sizes=chunk_sizes,
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HierarchicalNodeParser"

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: list[BaseNode],
        level: int,
        show_progress: bool = False,
    ) -> list[BaseNode]:
        """Recursively get nodes from nodes."""
        if level >= len(self.node_parser_ids):
            raise ValueError(
                f"Level {level} is greater than number of text "
                f"splitters ({len(self.node_parser_ids)})."
            )

        # first split current nodes into sub-nodes
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing documents into nodes"
        )
        sub_nodes = []
        for node in nodes_with_progress:
            cur_sub_nodes = self.node_parser_map[
                self.node_parser_ids[level]
            ].get_nodes_from_documents([node])
            # add parent relationship from sub node to parent node
            # add child relationship from parent node to sub node
            # relationships for the top-level document objects that            # NOTE: Only add relationships if level > 0, since we don't want to add we are splitting
            if level > 0:
                for sub_node in cur_sub_nodes:
                    _add_parent_child_relationship(
                        parent_node=node,
                        child_node=sub_node,
                    )

            sub_nodes.extend(cur_sub_nodes)

        # now for each sub-node, recursively split into sub-sub-nodes, and add
        if level < len(self.node_parser_ids) - 1:
            sub_sub_nodes = self._recursively_get_nodes_from_nodes(
                sub_nodes,
                level + 1,
                show_progress=show_progress,
            )
        else:
            sub_sub_nodes = []

        return sub_nodes + sub_sub_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: list[BaseNode] = []
            documents_with_progress = get_tqdm_iterable(
                documents, show_progress, "Parsing documents into nodes"
            )

            # TODO: a bit of a hack rn for tqdm
            for doc in documents_with_progress:
                nodes_from_doc = self._recursively_get_nodes_from_nodes([doc], 0)
                all_nodes.extend(nodes_from_doc)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        return list(nodes)

from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.schema import Document as LlamaIndexDocument
from llama_index.core.schema import TextNode
from loguru import logger

logger.add("logs/node_parsers.log")


def mutil_level_markdown_splitter(
    document: LlamaIndexDocument,
    primary_headers: list[tuple[str, str]] | None = None,
    secondary_headers: list[tuple[str, str]] | None = None,
    min_chapter_count: int = 5,
    max_chunk_size: int = 4000,
    strip_headers: bool = True,
) -> list[TextNode]:
    """处理Markdown文档并生成自适应分块的节点列表

    Args:
        document: LlamaIndex Document对象
        primary_headers: 主要分割标题级别，默认一级标题
        secondary_headers: 次要分割标题级别，在需要时使用
        min_chapter_count: 最小期望章节数量
        max_chunk_size: 单个节点最大字符数
        strip_headers: 是否从章节内容中移除标题

    Returns:
        TextNode列表
    """
    # 设置默认值
    primary_headers = primary_headers or [("#", "Header1")]
    secondary_headers = secondary_headers or [("#", "Header1"), ("##", "Header2")]

    # 获取文档内容和元数据
    text_content = document.get_content()
    base_metadata = document.metadata

    # 尝试使用主要标题级别分割
    try:
        primary_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=primary_headers, strip_headers=strip_headers
        )
        primary_chunks = primary_splitter.split_text(text_content)

        # 检查是否需要更细粒度的分割
        if len(primary_chunks) >= min_chapter_count:
            # 章节数量足够，使用主要分割
            logger.info(f"primary_chunks enough: {len(primary_chunks)}")
            return _convert_chunks_to_nodes(primary_chunks, base_metadata)

        # 检查是否存在过大的chunk
        large_chunks_exist = any(
            len(chunk.page_content) > max_chunk_size for chunk in primary_chunks
        )

        logger.info(f"large_chunks_exist: {large_chunks_exist}")

        if large_chunks_exist and len(primary_chunks) < min_chapter_count:
            # 尝试次要分割级别
            secondary_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=secondary_headers, strip_headers=strip_headers
            )
            secondary_chunks = secondary_splitter.split_text(text_content)

            # 如果次要分割增加了足够的粒度，使用它
            if len(secondary_chunks) >= min_chapter_count:
                return _convert_chunks_to_nodes(secondary_chunks, base_metadata)

            # 如果次要分割仍不够，尝试段落分割
            # 这里使用简单的分段逻辑，可根据需要替换为ChineseRecursiveTextSplitter等
            from readai.components.text_splitters.chinese_text_splitter import (
                ChineseRecursiveTextSplitter,
            )

            paragraph_splitter = ChineseRecursiveTextSplitter(
                chunk_size=128, chunk_overlap=20
            )
            paragraph_chunks = []

            # 如果二级标题分割后仍有大块，对每个块再次分割

            # TODO 这里可以建立父子节点关系，对textnode处理
            for chunk in secondary_chunks:
                if len(chunk.page_content) > max_chunk_size:
                    # 对大块进行进一步分割
                    sub_texts = paragraph_splitter.split_text(chunk.page_content)
                    # 为子块添加元数据
                    for i, sub_text in enumerate(sub_texts):
                        paragraph_chunks.append(
                            _create_langchain_document(
                                sub_text, {**chunk.metadata, "sub_chunk": i + 1}
                            )
                        )
                else:
                    paragraph_chunks.append(chunk)

            return _convert_chunks_to_nodes(paragraph_chunks, base_metadata)

        # 默认使用主要分割
        return _convert_chunks_to_nodes(primary_chunks, base_metadata)

    except Exception as e:
        print(f"Markdown解析失败，使用简单分割: {e}")
        # 回退到简单的段落分割
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
        )
        fallback_chunks = fallback_splitter.split_text(text_content)
        return [
            TextNode(text=chunk, metadata=base_metadata) for chunk in fallback_chunks
        ]


def _convert_chunks_to_nodes(
    chunks: list[LangChainDocument], base_metadata=None
) -> list[TextNode]:
    """将Langchain文档块转换为LlamaIndex节点"""
    base_metadata = base_metadata or {}
    nodes = []

    for i, chunk in enumerate(chunks):
        # 创建节点，合并元数据
        node = TextNode(
            text=chunk.page_content,
            metadata={**base_metadata, **chunk.metadata, "chunk_id": i},
        )
        nodes.append(node)

    return nodes


def _create_langchain_document(text, metadata=None):
    """创建Langchain Document对象"""
    return LangChainDocument(page_content=text, metadata=metadata or {})


if __name__ == "__main__":
    from llama_index.core.schema import Document

    path = "/Users/pegasus/workplace/mygits/readest-ai/readai-backend/readai/components/test.md"
    with open(path, encoding="utf-8") as f:
        markdown_content = f.read()

    # 创建LlamaIndex的Document对象
    document = Document(
        text=markdown_content, metadata={"author": "me", "date": "2025-01-01"}
    )

    nodes = mutil_level_markdown_splitter(
        document,
        min_chapter_count=8,
        max_chunk_size=1000,
        strip_headers=True,
    )
    print(f"文档被分割为 {len(nodes)} 个节点")
    for i, node in enumerate(nodes):
        print(f"\n节点 {i + 1}:")
        print(f"元数据: {node.metadata}")
        print(f"{node.text[:100]}...")
        print(len(node.text))

import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core import Document
from llama_index.core.schema import TextNode
from markitdown import MarkItDown

LINK_PATTERN = re.compile(r"!?\[.*?\]\(.*?\)")
EMPTY_HEADING_PATTERN = re.compile(r"^\s*#+\s*$")
METADATA_PATTERN = re.compile(r"\*\*(\w+):\*\*\s*(.+)")
HEADER_SPLIT_ON = [
    ("#", "Header1"),
]


def convert_to_markdown(input_file: Path) -> str:
    """将输入文件(.epub)转换为Markdown格式并生成同名文件(.md)。

    参数:
    input_file (str): 输入epub文件的路径。
    """
    md = MarkItDown()
    result = md.convert(input_file)
    output_path = str(input_file).replace(".epub", ".md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(result.text_content)
    print(f"文件已成功转换并保存至 {output_path}")
    return output_path


def extract_metadata_from_markdown(markdown_text: str) -> dict:
    pattern = r"\*\*(\w+):\*\*\s*(.+)"
    metadata = {}

    for match in re.finditer(pattern, markdown_text):
        key = match.group(1).strip().lower()  # e.g., title, authors
        value = match.group(2).strip()
        metadata[key] = value

    return metadata


def process_markdown_file(input_path: str, output_path: str) -> dict:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    with open(input_path, encoding="utf-8") as f_in:
        raw_md = f_in.read()

    # 提取元信息
    metadata = extract_metadata_from_markdown(raw_md)

    raw_md = re.sub(LINK_PATTERN, "", raw_md)
    lines = raw_md.splitlines()
    # 去掉前6行
    lines = lines[6:]

    # 去掉lines中空行为line为换行符或空行的行
    lines = [
        line
        for line in lines
        if line.strip() != "" and not EMPTY_HEADING_PATTERN.match(line)
    ]

    # 写入新文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))

    print(f"✅ 已清洗并保存新文件: {output_path}")
    return metadata, output_path


def preprocess_book(book_path: str) -> tuple[dict, str]:
    """预处理epub文件，转换为markdown文件，并清洗数据。

    参数:
    book_path (str): 输入epub文件的路径。
    """
    converted_path = None
    try:
        converted_path = convert_to_markdown(book_path)
    except Exception as e:
        print(f"转换失败: {e}")
        return {}

    cleaned_path = converted_path.replace(".md", "_cleaned.md")
    metadata = process_markdown_file(converted_path, cleaned_path)
    return metadata, cleaned_path


# 从epub转换后的markdown文件中获取节点
def markdown_nodes(doc: Document) -> list[TextNode]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_SPLIT_ON,
        # strip_headers=False,
    )

    # 使用 splitter 处理纯文本
    split_documents = md_splitter.split_text(doc.text)

    print("该文档总共分割为", len(split_documents), "个节点")
    # 然后将结果转为 LlamaIndex 的 Node 对象
    nodes = []
    for split_doc in split_documents:
        node = TextNode(
            text=split_doc.page_content,
            metadata={
                **doc.metadata,  # 保留原始元数据
                **split_doc.metadata,  # 添加标题元数据
            },
        )
        nodes.append(node)
    return nodes


def get_nodes_from_file(file_path: str, metadata: dict) -> list[TextNode]:
    with open(file_path) as f:
        document = f.read()
    llama_doc = Document(text=document, metadata=metadata)
    return markdown_nodes(llama_doc)


def get_nodes_from_string(text: str, metadata: dict) -> list[TextNode]:
    llama_doc = Document(text=text, metadata=metadata)
    return markdown_nodes(llama_doc)


if __name__ == "__main__":
    # 获取当前文件所在路径的上级目录
    current_dir = Path(__file__).parent.parent
    book_path = current_dir / "tests/data/renzhi.epub"
    metadata, gen_md_path = preprocess_book(str(book_path))
    print(metadata)
    print(gen_md_path)
    md_nodes = get_nodes_from_file(gen_md_path, metadata)

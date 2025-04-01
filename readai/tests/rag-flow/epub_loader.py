import re
from pathlib import Path

from markitdown import MarkItDown

LINK_PATTERN = re.compile(r"!?\[.*?\]\(.*?\)")
EMPTY_HEADING_PATTERN = re.compile(r"^\s*#+\s*$")
METADATA_PATTERN = re.compile(r"\*\*(\w+):\*\*\s*(.+)")


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

    with open(input_file, encoding="utf-8") as f_in:
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
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))

    print(f"✅ 已清洗并保存新文件: {output_path}")
    return metadata


def convert_to_markdown(input_file: str, output_path: str):
    """将输入文件转换为Markdown格式并保存到指定路径。

    参数:
    input_file (str): 输入epub文件的路径。
    output_path (str): 输出Markdown文件的路径。
    """
    md = MarkItDown()

    result = md.convert(input_file)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(result.text_content)
    print(f"文件已成功转换并保存至 {output_path}")


if __name__ == "__main__":
    # 测试
    book_path = "../data/非暴力沟通.epub"
    convert_to_markdown(book_path, "../data/comunication_pre.md")
    metadata = process_markdown_file(
        "../data/comunication_pre.md",
        "../data/comunication_cleaned.md",
    )

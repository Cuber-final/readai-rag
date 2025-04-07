from llama_index.core.schema import NodeWithScore, TextNode


def merge_strings(A, B):
    # 找到A的结尾和B的开头最长的匹配子串
    max_overlap = 0
    min_length = min(len(A), len(B))

    for i in range(1, min_length + 1):
        if A[-i:] == B[:i]:
            max_overlap = i

    # 合并A和B，去除重复部分
    merged_string = A + B[max_overlap:]
    return merged_string


def get_node_content(
    node: NodeWithScore,
    embed_type=0,
    nodes: list[TextNode] | None = None,
    nodeid2idx: dict | None = None,
) -> str:
    text: str = node.get_content()
    return text

import re

from langchain.docstore.document import Document


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    try:
        ratio = alpha_count / total_count
        return ratio < threshold
    except ZeroDivisionError:
        return False


def is_possible_title(
    text: str,
    title_max_word_length: int = 20,
    non_alpha_threshold: float = 0.5,
) -> bool:
    """检查文本是否符合标题的特征。

    参数
    ----------
    text : str
        要检查的输入文本
    title_max_word_length : int, 可选
        标题可以包含的最大字符数，默认为20
    non_alpha_threshold : float, 可选
        文本需要被视为标题的非字母字符的最大比例，默认为0.5
    """
    if not text or len(text) > title_max_word_length or text.isnumeric():
        return False

    if text[-1] in ["，", "。", "，", "。"]:
        return False

    # 文本中有标点符号，就不是title
    ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
    ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
    if ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    # 文本中数字的占比不能太高，否则不是title
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):  # noqa: SIM103
        return False

    return True


def zh_title_enhance(docs: list[Document]) -> list[Document]:
    title = None
    if len(docs) > 0:
        for doc in docs:
            if is_possible_title(doc.page_content):
                doc.metadata["category"] = "cn_Title"
                title = doc.page_content
            elif title:
                doc.page_content = f"下文与({title})有关。{doc.page_content}"
        return docs
    else:
        print("文件不存在")

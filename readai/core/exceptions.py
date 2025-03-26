class ReadAIException(Exception):
    """ReadAI自定义基础异常"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class BookNotFoundException(ReadAIException):
    """书籍未找到异常"""

    def __init__(self, book_hash: str):
        self.book_hash = book_hash
        message = f"书籍未找到: {book_hash}"
        super().__init__(message)


class DocumentLoadException(ReadAIException):
    """文档加载异常"""

    def __init__(self, file_path: str, error: str):
        self.file_path = file_path
        self.error = error
        message = f"文档加载失败: {file_path}, 错误: {error}"
        super().__init__(message)


class LLMException(ReadAIException):
    """LLM调用异常"""

    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingException(ReadAIException):
    """Embedding异常"""

    def __init__(self, message: str):
        super().__init__(message)

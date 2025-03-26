from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class ReadAIException(Exception):
    """ReadAI自定义基础异常"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class BookNotFoundException(ReadAIException):
    """书籍未找到异常"""
    def __init__(self, book_id: str):
        self.book_id = book_id
        message = f"书籍未找到: {book_id}"
        super().__init__(message)


class BookNotProcessedException(ReadAIException):
    """书籍未处理异常"""
    def __init__(self, book_id: str):
        self.book_id = book_id
        message = f"书籍尚未处理成向量: {book_id}"
        super().__init__(message)


class DocumentLoadException(ReadAIException):
    """文档加载异常"""
    def __init__(self, file_path: str, error: str):
        self.file_path = file_path
        self.error = error
        message = f"文档加载失败: {file_path}, 错误: {error}"
        super().__init__(message)


class VectorStoreException(ReadAIException):
    """向量存储异常"""
    def __init__(self, message: str):
        super().__init__(message)


class LLMException(ReadAIException):
    """LLM调用异常"""
    def __init__(self, message: str):
        super().__init__(message)


class EmbeddingException(ReadAIException):
    """Embedding异常"""
    def __init__(self, message: str):
        super().__init__(message)


def http_exception_handler(status_code: int, message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
    """创建HTTP异常"""
    return HTTPException(
        status_code=status_code,
        detail={
            "code": status_code,
            "message": message,
            "details": details
        }
    ) 
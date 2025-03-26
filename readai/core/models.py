from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class BookMetadata(BaseModel):
    """书籍元数据模型"""
    book_id: str  # 唯一标识
    title: str
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    language: Optional[str] = None
    file_path: str
    file_type: Literal["pdf", "epub", "mobi", "txt"] = "pdf"
    total_pages: Optional[int] = None
    processed: bool = False  # 是否已处理为向量


class Message(BaseModel):
    """聊天消息模型"""
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class ChatHistory(BaseModel):
    """聊天历史记录模型"""
    book_id: str
    messages: List[Message] = []


class ChatRequest(BaseModel):
    """聊天请求模型"""
    book_id: str
    message: str
    model_name: Optional[str] = None
    stream: bool = True
    history_count: Optional[int] = 3  # 仅使用最近N条历史记录，None表示使用全部


class ChatResponse(BaseModel):
    """非流式聊天响应模型"""
    book_id: str
    response: str
    sources: Optional[List[Dict[str, Any]]] = None


class BookUploadResponse(BaseModel):
    """书籍上传响应模型"""
    book_id: str
    title: str
    message: str
    status: Literal["success", "error", "pending"]


class BookListResponse(BaseModel):
    """书籍列表响应模型"""
    books: List[BookMetadata]
    total: int


class ErrorResponse(BaseModel):
    """错误响应模型"""
    code: int
    message: str
    details: Optional[Dict[str, Any]] = None


class ProcessRequest(BaseModel):
    """处理书籍向量化请求模型"""
    book_id: str


class ProcessResponse(BaseModel):
    """处理书籍向量化响应模型"""
    book_id: str
    status: Literal["success", "error", "processing"]
    message: str 
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ChatStatus(str, PyEnum):
    SUCCESS = "success"
    RETRYING = "retrying"
    STOP = "stop"


class FileType(str, PyEnum):
    EPUB = "epub"
    PDF = "pdf"
    MOBI = "mobi"
    TXT = "txt"


def get_file_type(suffix: str) -> FileType:
    """根据文件后缀名映射为FileType"""
    if suffix == ".epub":
        return FileType.EPUB
    elif suffix == ".pdf":
        return FileType.PDF
    elif suffix == ".mobi":
        return FileType.MOBI
    elif suffix == ".txt":
        return FileType.TXT
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


class BookMetadata(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=True)
    status = Column(Enum(ChatStatus), default=ChatStatus.SUCCESS)
    created_at = Column(DateTime, default=datetime.now)
    file_type = Column(Enum(FileType), default=FileType.EPUB)
    file_name = Column(String, nullable=False)
    processed = Column(Boolean, default=False)  # 向量化处理
    # 关系
    messages = relationship(
        "ChatMessage", back_populates="book", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    role = Column(String, nullable=False)  # user 或 assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # 关系
    book = relationship("Book", back_populates="messages")

from datetime import datetime
from typing import List
import json

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Book(Base):
    """书籍表"""
    __tablename__ = "books"
    
    id = Column(String(36), primary_key=True)  # 使用UUID作为主键
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=True)
    publisher = Column(String(255), nullable=True)
    publication_date = Column(String(20), nullable=True)
    language = Column(String(20), nullable=True)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    total_pages = Column(Integer, nullable=True)
    processed = Column(Boolean, default=False)  # 是否已处理为向量
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    chats = relationship("Chat", back_populates="book", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "book_id": self.id,
            "title": self.title,
            "author": self.author,
            "publisher": self.publisher,
            "publication_date": self.publication_date,
            "language": self.language,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "processed": self.processed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Chat(Base):
    """聊天会话表"""
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(String(36), ForeignKey("books.id"), nullable=False)
    messages_json = Column(Text, nullable=False, default="[]")  # JSON格式存储消息
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    book = relationship("Book", back_populates="chats")
    
    @property
    def messages(self) -> List[dict]:
        """获取消息列表"""
        try:
            return json.loads(self.messages_json)
        except:
            return []
    
    @messages.setter
    def messages(self, value: List[dict]):
        """设置消息列表"""
        self.messages_json = json.dumps(value, ensure_ascii=False)
    
    def add_message(self, role: str, content: str):
        """添加一条消息"""
        messages = self.messages
        messages.append({"role": role, "content": content})
        self.messages = messages
    
    def to_dict(self):
        return {
            "id": self.id,
            "book_id": self.book_id,
            "messages": self.messages,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        } 
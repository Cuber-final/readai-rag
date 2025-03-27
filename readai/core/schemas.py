from enum import Enum

from pydantic import BaseModel


class Book(BaseModel):
    id: int
    title: str
    author: str | None = None
    hash: str

    # 定义方法根据hash到数据库获取book_id
    def get_book_id(self, hash: str) -> int:
        # 从数据库table:books中根据hash值获取book_id并更新
        self.book_id = self.hash_to_id(hash)
        return 1

    def get_chat_history(self):
        # 从数据库table:chat_messages中根据book_id获取chat_history
        pass

    def hash_to_id(self, hash: str) -> int:
        # 从数据库table:books中根据hash值获取book_id
        return 1


class BookUploadResponse(BaseModel):
    """书籍上传响应模型"""

    message: str
    code: int
    book_id: str


# 定义常见的响应状态码
class HttpStatus(Enum):
    OK = 200
    ERROR = 500


class LLMmode(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI_LIKE = "openailike"
    MOCK_LLM = "mock_llm"

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# 加载环境变量
ENV_FILE = os.getenv("ENV_FILE", ".env")
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "ReadAI-Backend"
    DEBUG: bool = False
    API_PREFIX: str = "/api"

    # 数据库配置
    DATABASE_URL: str

    # 向量存储配置
    VECTOR_STORE_TYPE: Literal["chroma", "qdrant"] = "chroma"
    CHROMA_PERSIST_DIR: str = "./data/vectorstore/chroma"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "readai"

    # 模型配置
    EMBEDDING_MODEL_TYPE: Literal["huggingface", "ollama"] = "huggingface"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "deepseek-coder:7b"

    # 文档配置
    DOCUMENT_DIR: str = "./data/books"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ENV_FILE
        case_sensitive = True


# 创建设置实例
settings = Settings()


# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    directories = [
        settings.DOCUMENT_DIR,
        settings.CHROMA_PERSIST_DIR,
        Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent,
    ]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


# 程序启动时确保目录存在
ensure_directories()

# 后期可以采取读取yaml配置文件来获取设置项

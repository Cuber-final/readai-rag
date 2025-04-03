import os
import time
from pathlib import Path

import mlflow
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# 配置 loguru 输出到文件
log_path = Path("logs/test_embedding.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, level="INFO", encoding="utf-8")

project_root = Path(os.getenv("PROJECT_ROOT"))
data_path = project_root / "data"

# 设置环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
model_name = os.getenv("DEEPSEEK_MODEL")

# 设置MLflow
try:
    mlflow.set_tracking_uri("http://localhost:5001")
    # 检查连接
    mlflow.get_tracking_uri()
    mlflow.set_experiment("LlamaIndex-RAG")
    mlflow.llama_index.autolog()
    logger.info("成功连接到MLflow服务器")
except Exception as e:
    logger.error(f"MLflow初始化错误: {e}")


class RAGTest:
    def __init__(self):
        # 设置文档解析器
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)

        # 设置 embedding 模型
        Settings.embed_model = OllamaEmbedding(
            model_name="quentinz/bge-large-zh-v1.5", base_url="http://localhost:11434"
        )

        # 设置 LLM
        Settings.llm = DeepSeek(model=model_name, api_key=api_key)

        # 初始化 Qdrant 客户端 - 只指定一种连接方式
        self.client = QdrantClient(host="localhost", port=6333)

        # 指定集合名称
        self.collection_name = "books_test"

        # 创建向量存储
        self.vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.collection_name
        )

        # 创建存储上下文
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # 初始化索引
        self.index = None

    def load_or_create_index(self, documents):
        """检查是否存在索引"""
        try:
            # 检查集合是否存在
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name in collection_names:
                logger.info(f"集合 '{self.collection_name}' 已存在，加载现有索引。")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store, storage_context=self.storage_context
                )
            else:
                start_time = time.time()
                self.index = VectorStoreIndex.from_documents(
                    documents, storage_context=self.storage_context
                )
                build_time = time.time() - start_time
                logger.info(f"索引创建成功并已插入文档，耗时: {build_time:.2f}秒")
        except UnexpectedResponse as e:
            logger.error(f"检查集合时发生错误: {e}")
            raise

    def chat(self, query, chat_history=None):
        """进行对话"""
        try:
            # 创建聊天引擎
            from llama_index.core.memory import ChatMemoryBuffer

            memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
            if chat_history:
                for msg in chat_history:
                    # 将字典转换为ChatMessage对象
                    chat_msg = ChatMessage(role=msg["role"], content=msg["content"])
                    memory.put(chat_msg)

            # 记录查询开始时间
            chat_engine = self.index.as_chat_engine(
                chat_mode="context", memory=memory, verbose=True, temperature=1.3
            )

            # 非流式响应
            response = chat_engine.chat(query)
            return response

        except Exception as e:
            logger.error(f"对话失败: {e}")
            raise

    def reset_index(self):
        """重置索引"""
        self.client.delete_collection(self.collection_name)
        self.index = None


def main():
    book_dir = data_path / "books"

    # 加载指定文件夹下的文档
    documents = SimpleDirectoryReader(
        book_dir,
        required_exts=[".epub"],
        recursive=True,
    ).load_data()

    # 创建 RAG 测试实例
    rag_test = RAGTest()

    with mlflow.start_run(run_name="RAG-Test"):
        rag_test.load_or_create_index(documents)
        # 进行交互式对话
        history = []
        query = "你是谁？"
        # 单次询问
        try:
            response = rag_test.chat(query, history)
            # 更新对话历史
            print(response)
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

import os
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
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


# 设置环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
model_name = os.getenv("DEEPSEEK_MODEL")


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
                self.index = VectorStoreIndex.from_documents(
                    documents, storage_context=self.storage_context
                )
                logger.info("索引创建成功并已插入文档。")
        except UnexpectedResponse as e:
            logger.error(f"检查集合时发生错误: {e}")
            raise

    def chat(self, query, history=None):
        """进行对话"""
        try:
            # 创建聊天引擎

            from llama_index.core.memory import ChatMemoryBuffer

            memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
            if history:
                for msg in history:
                    memory.put(msg)

            chat_engine = self.index.as_chat_engine(
                chat_mode="context", memory=memory, verbose=True, temperature=1.3
            )

            # 获取响应
            # streaming_response = chat_engine.stream_chat(query)
            # for token in streaming_response.response_gen:
            #     print(token, end="")
            # 非流式响应
            response = chat_engine.chat(query)
            print(response)
            # 将模型输出打到日志中
            logger.info(f"模型输出: {response}")
        except Exception as e:
            logger.error(f"对话失败: {e}")
            raise

    def reset_index(self):
        """重置索引"""
        self.client.delete_collection(self.collection_name)
        self.index = None


def main():
    book_dir = "/Users/pegasus/workplace/mygits/readest-ai/readai-backend/data/books"

    # 加载指定文件夹下的文档
    documents = SimpleDirectoryReader(
        book_dir,
        required_exts=[".epub"],
        recursive=True,
    ).load_data()

    # 创建 RAG 测试实例
    rag_test = RAGTest()

    rag_test.load_or_create_index(documents)
    # 创建索引并插入文档

    # 进行交互式对话
    history = []

    query = input("\nquestion : ")
    try:
        rag_test.chat(query, history)
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

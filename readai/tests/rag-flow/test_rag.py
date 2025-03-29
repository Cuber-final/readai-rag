import logging
import os
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"  # 替换为您的API key

# 设置持久化存储路径
VECTOR_STORE_PATH = Path("./data/vectorstore")
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)


class RAGTest:
    def __init__(self):
        # 设置文档解析器
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        # 设置embedding模型
        Settings.embed_model = OllamaEmbedding(
            model_name="quentinz/bge-large-zh-v1.5", base_url="http://localhost:11434"
        )

        # 设置LLM
        Settings.llm = DeepSeek(
            model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY")
        )

        # 初始化Qdrant客户端，配置持久化存储
        self.client = QdrantClient(
            host="localhost",
            port=6333,
            path=str(VECTOR_STORE_PATH / "qdrant"),  # 设置持久化存储路径
        )

        # 创建向量存储
        self.vector_store = QdrantVectorStore(
            client=self.client, collection_name="test_collection"
        )

        # 创建存储上下文
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # 初始化索引
        self.index = None

    def create_index(self, documents: list[str]):
        """创建索引"""
        try:
            # 创建索引
            self.index = VectorStoreIndex.from_documents(
                documents, storage_context=self.storage_context
            )
            logger.info("索引创建成功")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise

    def chat(self, query: str, history: list[dict[str, str]] = None) -> str:
        """进行对话"""
        try:
            if not self.index:
                raise ValueError("请先创建索引")

            # 创建聊天引擎
            chat_engine = self.index.as_chat_engine(chat_mode="context", memory=history)

            # 获取响应
            response = chat_engine.chat(query)
            return response.response

        except Exception as e:
            logger.error(f"对话失败: {e}")
            raise

    def load_existing_index(self):
        """加载已存在的索引"""
        try:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            logger.info("成功加载已存在的索引")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False


def main():
    # 测试文档
    test_documents = [
        "人工智能（AI）正在深刻改变我们的生活方式。",
        "机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的特征。",
        "自然语言处理（NLP）是人工智能的一个重要应用领域，它使计算机能够理解和处理人类语言。",
    ]

    # 创建RAG测试实例
    rag_test = RAGTest()

    # 尝试加载已存在的索引，如果不存在则创建新索引
    if not rag_test.load_existing_index():
        rag_test.create_index(test_documents)

    # 测试对话
    history = []
    while True:
        query = input("\n请输入您的问题（输入'quit'退出）: ")
        if query.lower() == "quit":
            break

        try:
            response = rag_test.chat(query, history)
            print(f"\n回答: {response}")

            # 更新对话历史
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

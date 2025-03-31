import os
from pathlib import Path

import mlflow
from langchain_community.document_loaders import UnstructuredEPubLoader
from llama_index.core import Document
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient

from readai.components.custom.text_splitters.chinese_text_splitter import (
    ChineseRecursiveTextSplitter,
)

# 设置日志
log_path = Path("logs/test_embedding.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
logger.add(log_path, level="INFO", encoding="utf-8")

# 设置MLflow
try:
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.get_tracking_uri()
    mlflow.set_experiment("LlamaIndex-RAG")
    mlflow.llama_index.autolog()
    logger.info("成功连接到MLflow服务器")
except Exception as e:
    logger.error(f"MLflow初始化错误: {e}")

# 设置模型
api_key = os.getenv("DEEPSEEK_API_KEY")
model_name = os.getenv("DEEPSEEK_MODEL")
Settings.llm = DeepSeek(model=model_name, api_key=api_key)
Settings.embed_model = OllamaEmbedding(
    model_name="quentinz/bge-large-zh-v1.5", base_url="http://localhost:11434"
)

# 第一步：使用 UnstructuredEPubLoader 加载 EPUB 文档
test_data_path = Path(
    "/Users/pegasus/workplace/mygits/readest-ai/readai-backend/readai/tests/data"
)
book_name = "非暴力沟通.epub"
book_path = test_data_path / book_name
loader = UnstructuredEPubLoader(book_path, mode="elements", strategy="hi_res")
documents = loader.load()

# 使用中文文本分割器
text_splitter = ChineseRecursiveTextSplitter(chunk_size=1000, chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

# 转换为llamaindex需要的documents对象
llama_documents = [
    Document(text=doc.page_content, metadata=doc.metadata) for doc in split_documents
]

# 创建层级解析器
parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 1024])
book_nodes = parser.get_nodes_from_documents(llama_documents)
leaf_nodes = get_leaf_nodes(book_nodes)
root_nodes = get_root_nodes(book_nodes)

print(f"切分后节点总数量: {len(leaf_nodes)}")
print(f"根节点数量: {len(root_nodes)}")

# 初始化Qdrant客户端
qdrant_client = QdrantClient(
    host="localhost", port=6333
)  # 使用内存存储，也可以指定本地路径
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="book_test0331")

# 创建文档存储
docstore = SimpleDocumentStore()
docstore.add_documents(book_nodes)

# 创建存储上下文
storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

# 创建基础索引
base_index = VectorStoreIndex(nodes=leaf_nodes, storage_context=storage_context)

# 创建基础检索器
base_retriever = base_index.as_retriever(similarity_top_k=6)

# 创建AutoMergingRetriever
from llama_index.core.retrievers import AutoMergingRetriever

retriever = AutoMergingRetriever(
    base_retriever, storage_context, verbose=True, simple_ratio_thresh=0.6
)

# 创建查询引擎
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)

# 执行查询
query = "非暴力沟通的第一个要素是什么？"
response = query_engine.query(query)
print(response)

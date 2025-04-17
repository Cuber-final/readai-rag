# ReadAI-Backend

## 📖 AI 阅读助手（RAG-Powered eBook Chatbot）

---

一个基于电子书的智能阅读助手，支持上传 PDF/EPUB 文档，通过大语言模型进行内容理解与多轮问答。支持本地部署模型（如 Ollama）与云端 API。

---

## 🌟 项目特色

- 📚 支持 PDF、EPUB 格式电子书，自动加载并切分内容用于检索
- 🔍 基于 [LlamaIndex](https://www.llamaindex.ai/) 构建的 RAG 流程，支持检索，多轮对话
- 🧠 多模型对接：支持 Ollama 框架下可用的模型
- 💬 流式聊天：兼容 ChatGPT 风格的逐字输出体验（SSE 实现）
- 💡 自定义可拓展：前端选择模型，后端自动调度，配置灵活
- 🧱 技术栈：FastAPI + LlamaIndex + ChromaDB/Qdrant

---

## 🚀 快速开始

### 前置条件

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) 包管理工具
- [Ollama](https://ollama.com/download) 本地LLM框架


## 🧠 技术栈一览

| 模块 | 技术 |
| --- | --- |
| 后端框架 | FastAPI, Pydantic, SQLAlchemy |
| 向量存储 | ChromaDB, Qdrant |
| LLM框架 | Ollama |
| RAG引擎 | LlamaIndex |
| 文档处理 | PyPDF, EbookLib |
| 数据库 | SQLite |
| Embedding模型 | BAAI/bge-large-zh-v1.5 |

## 💻 开发计划
 - [ ] 统一的模型接口层LLM

### 自定义配置

主要配置项在 `app/core/config.py` 中定义，可通过环境变量或 `.env` 文件覆盖默认配置。
如果使用torch_gpu，需要修改pyproject.toml 中的torch依赖
torch = {version = "2.6.0+cu126", source = "torch",optional = true}

并在其中添加下载源
[[tool.poetry.source]]
name = "torch"
url = "https://download.pytorch.org/whl/cu126"
priority = "supplemental"

### 扩展功能

- 添加新的文档类型支持：扩展 `document_loader.py` 中的加载器
- 集成新的向量存储：扩展 `vector_store.py` 中的存储服务
- 添加新的LLM支持：修改 `llm.py` 中的模型接口

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
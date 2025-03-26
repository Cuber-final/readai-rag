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

### 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/readai-backend.git
cd readai-backend
```

2. 使用 Poetry 安装依赖

```bash
poetry install
```

3. 创建 `.env` 文件 (可复制 `.env.example` 并修改配置)

```bash
cp .env.example .env
```

4. 修改 `.env` 文件中的配置项，尤其是 Ollama 模型配置和数据存储路径

### 运行

1. 确保 Ollama 服务已启动并且已拉取所需模型

```bash
# 下载模型（以 deepseek-coder 为例）
ollama pull deepseek-coder:7b
```

2. 初始化数据库

```bash
poetry run python scripts/init_db.py
```

3. 启动应用

```bash
poetry run uvicorn app.main:app --reload
```

4. 访问文档以测试 API

打开浏览器访问 http://localhost:8000/docs

## 🧪 测试 RAG 功能

可以使用测试脚本快速测试 RAG 功能：

```bash
poetry run python scripts/test_rag.py /path/to/your/document.pdf "你的测试问题"
```

## 📚 API 文档

启动服务后，访问以下地址查看 API 文档:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要 API 端点

- `POST /api/documents/upload`: 上传 PDF 或 EPUB 电子书
- `POST /api/documents/process`: 处理文档生成向量索引
- `GET /api/documents/books`: 获取所有上传的书籍
- `POST /api/chat/chat`: 与书籍进行对话 (支持流式输出)
- `GET /api/chat/history/{book_id}`: 获取与书籍的聊天历史
- `DELETE /api/chat/history/{book_id}`: 清空与书籍的聊天历史

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

## 💻 开发指南

### 项目结构

```
readai-backend/
├── app/                    # 主应用代码
│   ├── api/                # API端点
│   │   └── endpoints/      # API实现
│   ├── core/               # 核心配置和工具
│   ├── db/                 # 数据库模型和会话
│   └── services/           # 业务服务
├── data/                   # 数据存储目录
├── scripts/                # 实用脚本
└── tests/                  # 测试代码
```

### 自定义配置

主要配置项在 `app/core/config.py` 中定义，可通过环境变量或 `.env` 文件覆盖默认配置。

### 扩展功能

- 添加新的文档类型支持：扩展 `document_loader.py` 中的加载器
- 集成新的向量存储：扩展 `vector_store.py` 中的存储服务
- 添加新的LLM支持：修改 `llm.py` 中的模型接口

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件
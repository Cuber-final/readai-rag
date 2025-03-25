## 📖 AI 阅读助手（RAG-Powered eBook Chatbot）

---

一个基于电子书的智能阅读助手，支持上传 PDF/EPUB 文档，通过大语言模型进行内容理解与多轮问答。支持本地部署模型（如 Ollama、vLLM）与云端 API（如 DeepSeek、OpenAI、Qwen）

**注：当前仓库为后端项目代码**

---

## 🌟 项目特色

- 📚 支持 PDF、EPUB 格式电子书，自动加载并切分内容用于检索
- 🔍 基于 [LlamaIndex](https://www.llamaindex.ai/) 构建的 RAG 流程，支持检索，多轮对话
- 🧠 多模型对接：支持 OpenAI、DeepSeek、Qwen 等 API 及本地 Ollama框架下可用的模型
- 💬 流式聊天：兼容 ChatGPT 风格的逐字输出体验（SSE 实现）
- 💡 自定义可拓展：前端选择模型，后端自动调度，配置灵活
- 🧱 技术栈：FastAPI + LlamaIndex + ChromaDB （后端）

---

## 🛠️ 开发任务清单（TODO）

### 📦 模型支持

- [ ]  云端模型调用api封装（OpenAI / DeepSeek / Qwen）
- [ ]  本地模型部署（Ollama）流式输出支持
- [ ]  模型配置统一管理，前端通过 `model_name` 控制选择

### 📚 文档与检索系统

- [ ]  支持 EPUB / PDF 文档加载与分块
- [ ]  向量化 & 存储（ChromaDB）
- [ ]  基于LLamaIndex实现的 RAG pipeline ，适配openai-like的LLM与生成响应封装

### 💬 FastAPI 接口

- [ ]  SSE 流式输出接口 `/chat`（前端逐字输出支持）
- [ ]  错误处理、token 控制、上下文限制
- [ ]  前端任务进度提示（文档同步中 / 向量库未构建等）

## 🧠 技术栈一览

| 模块 | 技术 |
| --- | --- |
| 前端 | Tauri, React, TailwindCSS, Zustand |
| 后端 | FastAPI, LlamaIndex, ChromaDB，Qdrant |
| LLM以及推理框架 | DeepSeek, Ollama |
| 构建 & 部署 | Poetry, Uvicorn |
| 状态管理 | SQLite |

---

## 🤝 灵感来自以下项目

- [Readest](https://github.com/ilyasotkov/readest)：本项目前端 UI 原型来源
- [LlamaIndex](https://www.llamaindex.ai/)：RAG 框架
- [ChromaDB](https://www.trychroma.com/)：向量数据库
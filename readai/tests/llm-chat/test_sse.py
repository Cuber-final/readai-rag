import asyncio
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse  # 使用专门的SSE响应类

# 创建FastAPI应用
app = FastAPI()

# 配置CORS中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，实际生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模拟GPT响应数据
SAMPLE_RESPONSES = [
    "你好，我是AI助手，很高兴为你服务！",
    "FastAPI是一个现代、快速的Python Web框架，用于构建API。",
    "SSE（Server-Sent Events）是一种服务器推送技术，允许服务器向客户端推送数据。",
]

# 设置SSE参数
STREAM_DELAY = 0.05  # 每字符延迟
RETRY_TIMEOUT = 5000  # 客户端重试超时(毫秒)


@app.get("/")
async def root():
    """首页路由"""
    return {"message": "欢迎使用FastAPI实现的SSE打字机效果API"}


@app.post("/chat")
async def chat(request: Request):
    """模拟聊天API
    接收用户消息并返回示例响应
    """
    data = await request.json()
    user_message = data.get("message", "")
    # 这里可以实现实际的处理逻辑，比如调用其他API或模型
    # 现在只是返回样本响应中的一个
    import random

    return {"response": random.choice(SAMPLE_RESPONSES)}


@app.get("/stream-chat")
async def stream_chat(request: Request, message: str = ""):
    """实现SSE流式聊天API
    逐字符返回响应，模拟打字机效果
    """

    # 设置事件生成器
    async def event_generator():
        try:
            # 选择一个示例响应
            import random

            response = random.choice(SAMPLE_RESPONSES)

            # 逐字符发送响应，模拟打字机效果
            for i in range(len(response) + 1):
                # 如果客户端断开连接，退出循环
                if await request.is_disconnected():
                    print("客户端已断开连接")
                    break

                # 构造当前需要发送的文本片段
                current_response = response[:i]

                # 使用事件字典格式，更符合SSE标准
                yield {
                    "event": "message",
                    "id": str(i),
                    "retry": RETRY_TIMEOUT,
                    "data": json.dumps(
                        {"text": current_response, "done": i == len(response)}
                    ),
                }

                # 添加延迟，使打字机效果更明显
                await asyncio.sleep(STREAM_DELAY)  # 每字符50毫秒
        except Exception as e:
            print(f"发生错误: {e}")
            # 发送错误消息
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    # 返回专门的SSE响应
    return EventSourceResponse(event_generator())


# 启动应用的入口点
if __name__ == "__main__":
    uvicorn.run("test_sse:app", host="127.0.0.1", port=8000, reload=True)

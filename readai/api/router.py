from fastapi import APIRouter

from readai.api.endpoints import chat, document

# 创建主路由
api_router = APIRouter()

# 添加聊天相关路由
api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["chat"]
)

# 添加文档相关路由
api_router.include_router(
    document.router,
    prefix="/documents",
    tags=["documents"]
) 
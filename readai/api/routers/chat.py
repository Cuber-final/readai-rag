import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from readai.core.schemas import HttpStatus
from readai.db.models import BookMetadata, ChatMessage
from readai.db.session import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/conversation/{book_id}")
async def chat(request: Request, db: Session = Depends(get_db)):
    # 从request中获取需要的变量
    request_data = await request.json()
    book_id = request_data.get("book_id")
    message = request_data.get("message")
    history_count = request_data.get("history_count")
    chat_mode = request_data.get("chat_mode")  # 表示采取RAG模式还是简单LLM多轮对话
    """传递book_id,接入LLM进行聊天(流式返回)"""
    try:
        # 结合当前书本的聊天记录,作为上下文给LLM
        chat_record = db.query(ChatMessage).filter(ChatMessage.book_id == book_id).all()

        # 如果没有聊天记录,创建一个新的
        if not chat_record:
            chat_record = ChatMessage(book_id=book_id, role="user", content=message)
            db.add(chat_record)
            db.commit()

        # 获取相关的所有message,封装为 {"role": "user", "content": "..."}
        history = [{"role": msg.role, "content": msg.content} for msg in chat_record]

        # 限制使用最近的N条历史记录
        if history_count is not None and history_count > 0:
            history = (
                history[-history_count * 2 :]
                if len(history) > history_count * 2
                else history
            )

        # 添加当前消息到对应book_id的历史记录
        chat_record = ChatMessage(book_id=book_id, role="user", content=message)
        db.add(chat_record)
        db.commit()

        response = ""
        if chat_mode == "RAG":
            # TODO 结合rag pipelne 返回非流式结果，以及流式方式输出
            pass
        else:
            # TODO 使用 query_engine 进行多轮对话，不涉及检索召回
            pass

        return {
            "code": HttpStatus.OK,
            "message": "success",
            "data": {
                "book_id": book_id,
                "content": response,
            },
        }

    except HTTPException as he:
        logger.info(f"发送聊天信息失败: {he!s}")
        return {
            "code": HttpStatus.ERROR,
            "message": "发送聊天信息失败",
            "data": {
                "error": str(he),  # 开发人员查看
            },
        }
    except Exception as e:
        logger.info(f"发送聊天信息失败: {e!s}")
        return {
            "code": HttpStatus.ERROR,
            "message": "发送聊天信息失败",  # 用户层不需要感知具体的异常信息
            "data": {
                "error": str(e),  # 开发人员查看
            },
        }


@router.get("/history/{book_id}")
async def get_chat_history(
    book_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(10, description="返回的消息数量限制"),
):
    """获取与书籍的聊天历史"""
    try:
        # 验证书籍是否存在
        book = db.query(BookMetadata).filter(BookMetadata.id == book_id).first()
        if not book:
            return {
                "code": HttpStatus.ERROR,
                "message": "书籍未找到",
                "data": {
                    "error": f"书籍未找到: {book_id}",
                },
            }

        # 获取聊天记录
        chat_record = db.query(ChatMessage).filter(ChatMessage.book_id == book_id).all()

        if not chat_record:
            # 如果没有聊天记录,返回空列表
            return {
                "code": HttpStatus.OK,
                "message": "success",
                "data": {"book_id": book_id, "messages": []},
            }

        # 获取消息,并限制数量
        history = [
            {"role": msg.role, "content": msg.get("conent", "")} for msg in chat_record
        ]
        if limit and len(history) > limit:
            history = history[-limit:]

        return {
            "code": HttpStatus.OK,
            "message": "success",
            "data": {"book_id": book_id, "history": history},
        }

    except HTTPException as e:
        logger.info(f"获取聊天历史失败: {e!s}")
        return {
            "code": HttpStatus.ERROR,
            "message": "获取聊天历史失败",
            "data": {
                "error": str(e),  # 开发人员查看
            },
        }
    except Exception as e:
        logger.info(f"获取聊天历史失败: {e!s}")
        return {
            "code": HttpStatus.ERROR,
            "message": "获取聊天历史失败",  # 用户层不需要感知具体的异常信息
            "data": {
                "error": str(e),  # 开发人员查看
            },
        }


@router.delete("/history/{book_id}")
async def clear_chat_history(book_id: str, db: Session = Depends(get_db)):
    """清空与书籍的聊天历史"""
    # TODO 待办
    pass

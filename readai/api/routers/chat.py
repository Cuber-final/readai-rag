import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from readai.core.schemas import ChatRequest, HttpStatus
from readai.db.models import BookMetadata, ChatMessage
from readai.db.session import get_db
from readai.services.rag_engine import rag_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/conversation/{book_id}")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """传递book_id,接入LLM进行聊天(流式返回)"""
    try:
        # 结合当前书本的聊天记录,作为上下文给LLM
        chat_record = (
            db.query(ChatMessage).filter(ChatMessage.book_id == request.book_id).all()
        )

        # 如果没有聊天记录,创建一个新的
        if not chat_record:
            chat_record = ChatMessage(
                book_id=request.book_id, role="user", content=request.message
            )
            db.add(chat_record)
            db.commit()

        # 获取相关的所有message,封装为 {"role": "user", "content": "..."}
        history = [{"role": msg.role, "content": msg.content} for msg in chat_record]

        # 限制使用最近的N条历史记录
        if request.history_count is not None and request.history_count > 0:
            history = (
                history[-request.history_count * 2 :]
                if len(history) > request.history_count * 2
                else history
            )

        # 添加当前消息到对应book_id的历史记录
        chat_record = ChatMessage(
            book_id=request.book_id, role="user", content=request.message
        )
        db.add(chat_record)
        db.commit()

        # 先实现非流式返回
        # response, sources = rag_engine.chat_with_book_no_stream(
        #     request.book_id, request.message, history, request.model_name
        # )

        # 先调用简单的llm-deepseek实现一个对话

        return {
            "code": HttpStatus.OK,
            "message": "success",
            "data": {
                "book_id": request.book_id,
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
    pass


async def generate_chat_stream(
    book_id: str,
    message: str,
    history: list[dict[str, str]],
    db: Session,
    model_name: str | None = None,
):
    """生成聊天流"""
    # SSE 格式头部
    yield "data: " + json.dumps({"status": "started"}) + "\n\n"

    try:
        full_response = ""

        # 通过RAG引擎获取响应
        async for token in rag_engine.chat_with_book(
            book_id, message, history, model_name
        ):
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
            # 适当的延迟，模拟打字效果
            await asyncio.sleep(0.01)

        # 响应完成,新增一条对应book_id的记录到chat_messages表
        chat_record = ChatMessage(
            book_id=book_id, role="assistant", content=full_response
        )
        db.add(chat_record)
        db.commit()

        # 结束标记
        yield "data: " + json.dumps({"status": "complete"}) + "\n\n"

    except Exception as e:
        logger.error(f"生成聊天流失败: {e!s}")
        # 发送错误信息
        error_message = f"错误: {e!s}"
        yield f"data: {json.dumps({'token': error_message})}\n\n"
        yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"

        # 尝试记录错误响应
        try:
            chat_record = ChatMessage(
                book_id=book_id, role="assistant", content=error_message
            )
            db.add(chat_record)
            db.commit()
        except Exception as e:
            logger.error(f"记录错误响应失败: {e!s}")
            return {
                "code": HttpStatus.ERROR,
                "message": "记录错误响应失败",
                "data": {
                    "error": str(e),  # 开发人员查看
                },
            }

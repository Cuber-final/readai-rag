from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from readai.core.models import ChatRequest, ChatResponse, Message, ChatHistory
from readai.core.exceptions import http_exception_handler, BookNotFoundException, BookNotProcessedException
from readai.db.session import get_db
from readai.db.models import Book, Chat
from readai.services.rag_engine import rag_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """与书籍聊天（流式返回）"""
    try:
        # 验证书籍是否存在
        book = db.query(Book).filter(Book.id == request.book_id).first()
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {request.book_id}")
        
        # 验证书籍是否已处理
        if not book.processed:
            raise http_exception_handler(400, f"书籍尚未处理为向量形式: {request.book_id}")
        
        # 获取历史聊天记录
        chat_record = db.query(Chat).filter(Chat.book_id == request.book_id).first()
        
        # 如果没有聊天记录，创建一个新的
        if not chat_record:
            chat_record = Chat(book_id=request.book_id, messages=[])
            db.add(chat_record)
            db.commit()
        
        # 获取历史消息
        history = chat_record.messages
        
        # 限制使用最近的N条历史记录
        if request.history_count is not None and request.history_count > 0:
            history = history[-request.history_count * 2:] if len(history) > request.history_count * 2 else history
        
        # 添加当前消息到历史记录
        chat_record.add_message("user", request.message)
        db.commit()
        
        if request.stream:
            # 流式返回
            return StreamingResponse(
                generate_chat_stream(request.book_id, request.message, history, chat_record.id, db, request.model_name),
                media_type="text/event-stream"
            )
        else:
            # 非流式返回
            response, sources = rag_engine.chat_with_book_no_stream(
                request.book_id,
                request.message,
                history,
                request.model_name
            )
            
            # 添加响应到历史记录
            chat_record.add_message("assistant", response)
            db.commit()
            
            return ChatResponse(
                book_id=request.book_id,
                response=response,
                sources=sources
            )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"聊天失败: {str(e)}")
        raise http_exception_handler(500, f"聊天失败: {str(e)}")


@router.get("/history/{book_id}")
async def get_chat_history(
    book_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(50, description="返回的消息数量限制")
):
    """获取与书籍的聊天历史"""
    try:
        # 验证书籍是否存在
        book = db.query(Book).filter(Book.id == book_id).first()
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {book_id}")
        
        # 获取聊天记录
        chat_record = db.query(Chat).filter(Chat.book_id == book_id).first()
        
        if not chat_record:
            # 如果没有聊天记录，返回空列表
            return ChatHistory(book_id=book_id, messages=[])
        
        # 获取消息，并限制数量
        messages = chat_record.messages
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        # 转换为响应格式
        message_list = []
        for msg in messages:
            message_list.append(Message(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        return ChatHistory(book_id=book_id, messages=message_list)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"获取聊天历史失败: {str(e)}")
        raise http_exception_handler(500, f"获取聊天历史失败: {str(e)}")


@router.delete("/history/{book_id}")
async def clear_chat_history(book_id: str, db: Session = Depends(get_db)):
    """清空与书籍的聊天历史"""
    try:
        # 验证书籍是否存在
        book = db.query(Book).filter(Book.id == book_id).first()
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {book_id}")
        
        # 获取聊天记录
        chat_record = db.query(Chat).filter(Chat.book_id == book_id).first()
        
        if chat_record:
            # 清空消息
            chat_record.messages = []
            db.commit()
        
        return {"message": "聊天历史已清空"}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"清空聊天历史失败: {str(e)}")
        raise http_exception_handler(500, f"清空聊天历史失败: {str(e)}")


async def generate_chat_stream(
    book_id: str,
    message: str,
    history: List[Dict[str, str]],
    chat_id: int,
    db: Session,
    model_name: Optional[str] = None
):
    """生成聊天流"""
    
    # SSE 格式头部
    yield "data: " + json.dumps({"status": "started"}) + "\n\n"
    
    try:
        full_response = ""
        
        # 通过RAG引擎获取响应
        async for token in rag_engine.chat_with_book(book_id, message, history, model_name):
            full_response += token
            yield f"data: {json.dumps({'token': token})}\n\n"
            # 适当的延迟，模拟打字效果
            await asyncio.sleep(0.01)
        
        # 响应完成，更新聊天记录
        chat_record = db.query(Chat).filter(Chat.id == chat_id).first()
        if chat_record:
            chat_record.add_message("assistant", full_response)
            db.commit()
        
        # 结束标记
        yield "data: " + json.dumps({"status": "complete"}) + "\n\n"
        
    except Exception as e:
        logger.error(f"生成聊天流失败: {str(e)}")
        # 发送错误信息
        error_message = f"错误: {str(e)}"
        yield f"data: {json.dumps({'token': error_message})}\n\n"
        yield "data: " + json.dumps({"status": "error", "message": str(e)}) + "\n\n"
        
        # 尝试记录错误响应
        try:
            chat_record = db.query(Chat).filter(Chat.id == chat_id).first()
            if chat_record:
                chat_record.add_message("assistant", error_message)
                db.commit()
        except:
            pass 
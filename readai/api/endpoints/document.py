from typing import List, Optional
import os
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query, Body
from sqlalchemy.orm import Session

from readai.core.models import BookMetadata, BookUploadResponse, BookListResponse, ProcessRequest, ProcessResponse
from readai.core.exceptions import DocumentLoadException, http_exception_handler
from readai.core.config import settings
from readai.db.session import get_db
from readai.db.models import Book
from readai.services.document_loader import document_loader
from readai.services.rag_engine import rag_engine
from readai.services.vector_store import vector_store_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=BookUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = Query(False, description="是否立即处理文档"),
    db: Session = Depends(get_db)
):
    """上传文档"""
    try:
        # 检查文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf", ".epub"]:
            raise http_exception_handler(
                400, f"不支持的文件类型: {file_ext}，仅支持PDF和EPUB格式"
            )
        
        # 保存文件
        book_id = str(uuid.uuid4())
        file_path = os.path.join(settings.DOCUMENT_DIR, f"{book_id}{file_ext}")
        
        # 确保目录存在
        os.makedirs(settings.DOCUMENT_DIR, exist_ok=True)
        
        # 写入文件
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 提取元数据
        metadata = document_loader.extract_metadata(file_path)
        metadata.book_id = book_id
        
        # 保存到数据库
        book = Book(
            id=book_id,
            title=metadata.title,
            author=metadata.author,
            publisher=metadata.publisher,
            publication_date=metadata.publication_date,
            language=metadata.language,
            file_path=file_path,
            file_type=metadata.file_type,
            processed=False
        )
        
        db.add(book)
        db.commit()
        
        # 根据用户选择是否立即处理文档
        if process_now:
            background_tasks.add_task(process_document_task, book_id, db)
            message = "文档上传成功，正在后台处理中"
            status = "pending"
        else:
            message = "文档上传成功，可稍后处理生成向量索引"
            status = "success"
        
        return BookUploadResponse(
            book_id=book_id,
            title=metadata.title,
            message=message,
            status=status
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise http_exception_handler(500, f"上传文档失败: {str(e)}")


@router.get("/books", response_model=BookListResponse)
async def get_books(
    db: Session = Depends(get_db),
    skip: int = Query(0, description="分页起始位置"),
    limit: int = Query(100, description="每页数量")
):
    """获取所有书籍"""
    try:
        # 查询所有书籍
        books = db.query(Book).offset(skip).limit(limit).all()
        total = db.query(Book).count()
        
        # 转换为响应模型
        book_list = []
        for book in books:
            book_list.append(BookMetadata(
                book_id=book.id,
                title=book.title,
                author=book.author,
                publisher=book.publisher,
                publication_date=book.publication_date,
                language=book.language,
                file_path=book.file_path,
                file_type=book.file_type,
                processed=book.processed
            ))
        
        return BookListResponse(books=book_list, total=total)
    
    except Exception as e:
        logger.error(f"获取书籍列表失败: {str(e)}")
        raise http_exception_handler(500, f"获取书籍列表失败: {str(e)}")


@router.get("/books/{book_id}", response_model=BookMetadata)
async def get_book(book_id: str, db: Session = Depends(get_db)):
    """获取书籍详情"""
    try:
        # 查询书籍
        book = db.query(Book).filter(Book.id == book_id).first()
        
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {book_id}")
        
        # 转换为响应模型
        return BookMetadata(
            book_id=book.id,
            title=book.title,
            author=book.author,
            publisher=book.publisher,
            publication_date=book.publication_date,
            language=book.language,
            file_path=book.file_path,
            file_type=book.file_type,
            processed=book.processed
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"获取书籍详情失败: {str(e)}")
        raise http_exception_handler(500, f"获取书籍详情失败: {str(e)}")


@router.post("/process", response_model=ProcessResponse)
async def process_document(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """处理文档（生成向量索引）"""
    try:
        # 查询书籍
        book = db.query(Book).filter(Book.id == request.book_id).first()
        
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {request.book_id}")
        
        # 检查文件是否存在
        if not os.path.exists(book.file_path):
            raise http_exception_handler(404, f"文件不存在: {book.file_path}")
        
        # 检查是否已处理
        if book.processed:
            return ProcessResponse(
                book_id=book.id,
                status="success",
                message="文档已处理过，无需重新处理"
            )
        
        # 在后台处理文档
        background_tasks.add_task(process_document_task, book.id, db)
        
        return ProcessResponse(
            book_id=book.id,
            status="processing",
            message="文档处理任务已启动，正在后台处理中"
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"处理文档失败: {str(e)}")
        raise http_exception_handler(500, f"处理文档失败: {str(e)}")


@router.delete("/books/{book_id}")
async def delete_book(book_id: str, db: Session = Depends(get_db)):
    """删除书籍"""
    try:
        # 查询书籍
        book = db.query(Book).filter(Book.id == book_id).first()
        
        if not book:
            raise http_exception_handler(404, f"书籍未找到: {book_id}")
        
        # 删除文件
        try:
            if os.path.exists(book.file_path):
                os.remove(book.file_path)
        except Exception as e:
            logger.warning(f"删除文件失败(继续删除数据库记录): {str(e)}")
        
        # 删除向量索引
        try:
            vector_store_service.delete_index(book_id)
        except Exception as e:
            logger.warning(f"删除向量索引失败(继续删除数据库记录): {str(e)}")
        
        # 删除数据库记录
        db.delete(book)
        db.commit()
        
        return {"message": "书籍删除成功"}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"删除书籍失败: {str(e)}")
        raise http_exception_handler(500, f"删除书籍失败: {str(e)}")


async def process_document_task(book_id: str, db: Session):
    """处理文档任务（在后台运行）"""
    try:
        # 查询书籍
        book = db.query(Book).filter(Book.id == book_id).first()
        
        if not book:
            logger.error(f"处理文档任务: 书籍未找到: {book_id}")
            return
        
        # 处理书籍
        result = await rag_engine.process_book(book)
        
        if result:
            # 更新处理状态
            book.processed = True
            db.commit()
            logger.info(f"处理文档任务完成: {book_id}")
        else:
            logger.error(f"处理文档任务失败: {book_id}")
    
    except Exception as e:
        logger.error(f"处理文档任务异常: {str(e)}")
        # 确保数据库会话被关闭
        db.close() 
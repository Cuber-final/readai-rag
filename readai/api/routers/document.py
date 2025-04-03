import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy.orm import Session

from readai.components.loaders import document_loader
from readai.core.config import settings
from readai.core.schemas import (
    BookUploadResponse,
    HttpStatus,
)
from readai.db.models import BookMetadata, ChatStatus, get_file_type
from readai.db.session import get_db
from readai.services.rag_engine import rag_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=BookUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = Query(False, description="是否立即处理文档"),
    db: Session = Depends(get_db),
):
    """上传文档"""
    try:
        # 检查文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".pdf", ".epub", ".mobi", ".txt"]:
            return BookUploadResponse(
                book_id=None,
                message=f"不支持的文件类型: {file_ext}, 支持的文件类型: .pdf, .epub, .mobi, .txt",
                code=HttpStatus.ERROR,
            )

        # 保存文件,生成file_name+时间戳的唯一命名
        original_name = Path(file.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        save_file_name = f"{original_name}_{timestamp}{file_ext}"
        file_path = os.path.join(settings.DOCUMENT_DIR, save_file_name)

        # 确保目录存在
        os.makedirs(os.path.join(settings.DOCUMENT_DIR), exist_ok=True)

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 提取元数据
        metadata = document_loader.extract_metadata(file_path)

        # 保存到数据库,创建时自动生成book_id
        book = BookMetadata(
            title=metadata.title,
            author=metadata.author,
            file_name=save_file_name,
            file_type=get_file_type(file_ext),
            status=ChatStatus.SUCCESS,
        )

        db.add(book)
        db.commit()
        book_id = book.id

        # 根据用户选择是否立即处理文档
        if process_now:
            background_tasks.add_task(process_document_task, book_id, db)
            message = "文档上传成功,正在后台处理中"
        else:
            message = "文档上传成功,可稍后处理生成向量索引"

        return BookUploadResponse(book_id=book_id, message=message, code=HttpStatus.OK)

    except HTTPException as e:
        raise e
    except Exception:
        logger.error("上传文档失败")
        return BookUploadResponse(
            book_id=None, message="上传文档失败", code=HttpStatus.ERROR
        )


@router.delete("/books/{book_id}")
async def delete_book(book_id: str, db: Session = Depends(get_db)):
    """删除书籍"""
    # TODO:根据前端触发删除,传入book_id,到数据库中删除
    pass


async def process_document_task(book_id: str, db: Session):
    """处理文档任务(在后台运行)"""
    try:
        # 查询书籍
        book = db.query(BookMetadata).filter(BookMetadata.id == book_id).first()

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
            # 默认处理状态是false所以不需要处理
            logger.error(f"处理文档任务失败: {book_id}")

    except Exception as e:
        logger.error(f"处理文档任务异常: {e!s}")
        # 确保数据库会话被关闭
        db.close()

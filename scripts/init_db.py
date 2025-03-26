#!/usr/bin/env python
"""
初始化数据库脚本
执行此脚本将创建必要的数据库表
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from readai.db.session import init_db
from readai.core.config import ensure_directories

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # 确保目录存在
        ensure_directories()
        
        # 初始化数据库
        logger.info("正在初始化数据库...")
        init_db()
        logger.info("数据库初始化完成")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}", exc_info=True)
        sys.exit(1) 
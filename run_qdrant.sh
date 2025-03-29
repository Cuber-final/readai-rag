#!/bin/bash

# 获取当前脚本所在目录的绝对路径
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 定义数据存储路径
DATA_PATH="$PROJECT_ROOT/data/vectorstore/qdrant"

#  输出检查DATA_PAT
echo "DATA_PATH: $DATA_PATH"

# 创建数据存储目录（如果不存在）
mkdir -p "$DATA_PATH"

# 启动 Qdrant 容器
docker run -d \
    --name qdrant \
    -p 6333:6333 \
    -v "$DATA_PATH:/qdrant/storage" \
    qdrant/qdrant
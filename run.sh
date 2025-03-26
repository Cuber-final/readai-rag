 #!/bin/bash

# 确保Python环境存在
if command -v poetry &> /dev/null
then
    echo "Poetry 已安装，开始启动应用..."
else
    echo "需要先安装 Poetry 包管理工具"
    echo "请访问: https://python-poetry.org/docs/#installation"
    exit 1
fi

# 安装依赖
echo "安装项目依赖..."
poetry install

# 确保数据库初始化
echo "初始化数据库..."
poetry run python scripts/init_db.py

# 启动应用
echo "启动应用服务..."
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 如果应用终止，显示提示
echo "应用已停止运行"
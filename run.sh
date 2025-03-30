 #!/bin/bash

# 创建适配版本的虚拟环境
pyenv virtualenv 3.10.16 read_backend

# 激活虚拟环境
pyenv activate read_backend

# 通过poetry安装依赖
poetry env use $(pyenv which python3)
poetry install

# 确保数据库初始化
echo "初始化数据库..."
poetry run python scripts/init_db.py

# 启动应用
echo "启动应用服务..."
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 如果应用终止，显示提示
echo "应用已停止运行"
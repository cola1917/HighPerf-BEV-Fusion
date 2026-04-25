# 使用轻量级且科学计算友好的基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统级依赖
# build-essential: 编译某些 Python 库
# libgl1 & libglib2.0: OpenCV 运行必备
# procps: 用于监控进程性能
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装核心性能库
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas \
    matplotlib \
    opencv-python-headless \
    nuscenes-devkit \
    pyquaternion \
    aiofiles \
    numba \
    umap-learn

# 设置环境变量，确保 Python 输出实时刷新到终端
ENV PYTHONUNBUFFERED=1

# 默认进入 python 交互界面
CMD ["python"]
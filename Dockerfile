# 基于 PyTorch CUDA 镜像
FROM aqtbrodh2awvkc.xuanyuan.run/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# 设置中国镜像源（可选，根据网络环境决定是否启用）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml .

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 复制源代码
COPY src/ ./src/
COPY scripts/ ./scripts/

# 创建数据目录
RUN mkdir -p data/pdfs data/parsed data/qdrant

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# C-Former GovLayer 服务镜像
# 设计：治理层纯标准库，不依赖 torch —— 镜像小（~150MB）、启动秒级
FROM python:3.11-slim

WORKDIR /app

# 仅装服务依赖（不装 torch/numpy，训练相关依赖不进部署镜像）
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 复制治理内核 + 知识库数据 + 服务与前端
COPY cformer_v63/ ./cformer_v63/
COPY data/ ./data/
COPY server/ ./server/

ENV PYTHONUNBUFFERED=1
# 可选：设置后可启用 LLM 自然语言生成（只喂权限过滤后的可见内容）
# ENV DEEPSEEK_API_KEY=sk-xxx

EXPOSE 8000
WORKDIR /app/server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

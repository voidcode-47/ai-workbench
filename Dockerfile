FROM python:3.11-slim

WORKDIR /app

# 系统依赖（python-docx / pypdf 均为纯 Python，无需额外库）
COPY requirements.txt .
# 使用清华 PyPI 镜像加速依赖下载（海外源在国内网络下极慢）；
# 使用阿里云 PyPI 镜像加速依赖下载（海外源在国内网络下极慢，清华源 2026-09 起 403）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --default-timeout=120 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

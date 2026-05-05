FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./
RUN npm run build


FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY api.py ./
COPY services ./services
COPY frontend ./frontend

COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 10000

ENV API_HOST=0.0.0.0
ENV API_PORT=10000
ENV PYTHONUNBUFFERED=1

CMD ["python", "api.py"]

# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen

COPY src ./src
COPY data ./data
COPY web ./web
COPY db ./db
COPY .env.example ./

RUN mkdir -p /app/output /app/data/snapshots

CMD ["uv", "run", "python", "-m", "src.jpx_alpha_engine.run_scheduler"]

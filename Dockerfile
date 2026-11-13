FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY afrorisk ./afrorisk
COPY models/served ./models/served
COPY reports/metrics.json ./reports/metrics.json

EXPOSE 7860

CMD ["sh", "-c", "uvicorn afrorisk.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

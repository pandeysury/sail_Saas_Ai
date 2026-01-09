FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

RUN mkdir -p /app/data

# 👇 CHANGE 1
EXPOSE 8000

# 👇 CHANGE 2
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

# 👇 CHANGE 3 (MOST IMPORTANT)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
# COPY app/ ./app/
# # COPY static/ ./static/
# #COPY main.py .
# RUN mkdir -p /app/data
# EXPOSE 8000
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
# # CMD ["python", "main.py"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# # ---------- 1. Build React Frontend ----------
# FROM node:20-alpine AS frontend
# WORKDIR /frontend
# COPY frontend/chat-to-sms/package*.json ./
# RUN npm install
# COPY frontend/chat-to-sms/ .
# RUN npm run build


# # ---------- 2. Build FastAPI Backend ----------
# FROM python:3.11-slim AS backend
# WORKDIR /app

# # System dependencies
# RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# # Install Python requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# # Copy backend source
# COPY app/ ./app/

# # ✅ Copy built React app into static directory for FastAPI
# COPY --from=frontend /frontend/dist ./static

# # Create data folder
# RUN mkdir -p /app/data

# # Expose FastAPI port
# EXPOSE 8000

# # Healthcheck for Docker
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# # Run FastAPI app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.11-slim AS backend

# WORKDIR /app

# # System dependencies
# RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

# # Install Python packages first
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# # Copy backend source
# COPY app/ ./app/

# # Create data folder
# RUN mkdir -p /app/data

# EXPOSE 8000

# # Healthcheck
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# # Run server
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



# Stage 1: Build React frontend
FROM node:20 AS build-frontend
WORKDIR /build
COPY frontend/chat-to-sms/package*.json ./
RUN npm install
COPY frontend/chat-to-sms/ ./
RUN npm run build

# Stage 2: Backend
FROM python:3.11-slim AS backend
WORKDIR /app
# System dependencies
RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*
# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
# Copy backend source
COPY app/ ./app/
# Copy React build artifacts to /app/static
COPY --from=build-frontend /build/dist /app/static
# Create data folder
RUN mkdir -p /app/data
EXPOSE 8000
# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
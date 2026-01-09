from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import os

from app.config import settings
from app.state import get_bundle
from app.routers import query as query_router
from app.routers import chat as chat_router
from app.routers import feedback as feedback_router
from app.routers import dashboard as dashboard_router

app = FastAPI(title="RAG Hybrid API", version="1.0.0")

origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoints
@app.get("/health")
@app.get("/healthz")
def health_check():
    return {"status": "healthy", "service": "RAG Hybrid API"}

@app.get("/readyz")
def readiness_check():
    try:
        bundle = get_bundle()
        if bundle and hasattr(bundle, 'retriever') and bundle.retriever:
            return {"status": "ready", "retriever_loaded": True}
        else:
            return {"status": "not_ready", "retriever_loaded": False}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}

# Dynamic Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mount static
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Routers
app.include_router(query_router.router)
app.include_router(query_router.router, prefix="/{client_id}")
app.include_router(chat_router.router)
app.include_router(chat_router.router, prefix="/{client_id}")
app.include_router(feedback_router.router)
app.include_router(dashboard_router.router)

@app.get("/", response_class=HTMLResponse)
def serve_root():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

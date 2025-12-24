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
from app.routers import chat as chat_router

app = FastAPI(title="RAG Hybrid API", version="1.0.0")

origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Dynamic Paths (NO HARD-CODED PATHS) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]     # /app/
STATIC_DIR = PROJECT_ROOT / "static"                   # /app/static/
DATA_DIR = PROJECT_ROOT / "data"                       # /app/data/
DATA_DIR.mkdir(parents=True, exist_ok=True)            # auto-create

# ---------- Mount static ----------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------- Routers ----------
app.include_router(query_router.router)
app.include_router(query_router.router, prefix="/{client_id}")
app.include_router(chat_router.router)
app.include_router(chat_router.router, prefix="/{client_id}")


# ---------- Serve tenant documents (dynamic) ----------
@app.get("/{client_id}/docs/{filename:path}")
def serve_document(client_id: str, filename: str):
    """
    Serve documents from: /app/data/<client_id>/documents/<filename>
    """
    # Convert client_id to lowercase for consistent file paths
    client_id = client_id.lower()
    
    client_docs_dir = (DATA_DIR / client_id / "documents").resolve()
    full_path = (client_docs_dir / filename).resolve()

    # Security check (prevent escaping folder)
    if not str(full_path).startswith(str(client_docs_dir)):
        logger.error(f"Security violation attempt: {full_path}")
        raise HTTPException(status_code=404, detail="Invalid document path")

    if not full_path.is_file():
        logger.error(f"Document not found: {full_path}")
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info(f"Serving document: {full_path}")
    return FileResponse(str(full_path))


# ---------- SPA index ----------
def _index_file() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/", response_class=HTMLResponse)
def serve_root():
    return _index_file()


# ---------- SPA catch-all ----------
@app.get("/{full_path:path}")
async def serve_spa_catchall(full_path: str):
    clean_path = full_path.rstrip('/')
    parts = clean_path.split('/')
    filename = parts[-1] if parts else ''

    # Static file request?
    if filename and '.' in filename:
        file_path = STATIC_DIR / filename
        if file_path.is_file():
            logger.info(f"Serving static file: {file_path}")
            return FileResponse(file_path)

        file_path = STATIC_DIR / clean_path
        if file_path.is_file():
            logger.info(f"Serving static file: {file_path}")
            return FileResponse(file_path)

    # SPA route → return index.html
    logger.info(f"Serving SPA index.html for: /{full_path}")
    return _index_file()

from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse as StarletteFileResponse
from loguru import logger
import os

from app.config import settings
from app.state import get_bundle
from app.routers import query as query_router

app = FastAPI(title="RAG Hybrid API", version="1.0.0")

origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"

# Documents base path - use BASE_DIR from environment (matches indexer)
#DOCS_BASE_PATH = Path(os.getenv("BASE_DIR", r"C:\sms"))
DOCS_BASE_PATH = Path("/home/adminpc/Desktop/RAG_fastApi101025_FIXED/app/data")

# ---------- Mount static (OPTIONAL - only if you want /static/ prefix) ----------
# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# ---------- Routers ----------
app.include_router(query_router.router)
app.include_router(query_router.router, prefix="/{client_id}")



# ---------- Serve tenant documents ----------
@app.get("/{client_id}/docs/{filename:path}")
def serve_document(client_id: str, filename: str):
    """
    Serves HTML docs from BASE_DIR/<client_id>/documents/<filename>.
    """
    base = (DOCS_BASE_PATH / client_id / "documents").resolve()
    full = (base / filename).resolve()
    
    if not str(full).startswith(str(base)) or not full.is_file():
        logger.error(f"Document not found: {full}")
        raise HTTPException(status_code=404, detail="Document not found")
    
    logger.info(f"Serving: {full}")
    return FileResponse(str(full))

# ---------- SPA routes (MUST BE LAST) ----------
def _index_file() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

@app.get("/", response_class=HTMLResponse)
def serve_root():
    """Serve index.html at root"""
    return _index_file()

@app.get("/{full_path:path}")
async def serve_spa_catchall(full_path: str):
    """
    Smart static file server + SPA router:
    1. If requesting an actual file (script.js, style.css) → serve it
    2. If requesting a client route (/andriki) → serve index.html
    3. Handle both /andriki and /andriki/
    """
    # Remove trailing slash for consistency
    clean_path = full_path.rstrip('/')
    
    # Try to find the actual file first
    # Handle cases like:
    # - /script.js → serve script.js
    # - /andriki/script.js → serve script.js (from root)
    # - /style.css → serve style.css
    
    # Get the last segment (filename)
    parts = clean_path.split('/')
    filename = parts[-1] if parts else ''
    
    # Check if it's a static file request
    if filename and '.' in filename:
        # Try root level first
        file_path = STATIC_DIR / filename
        if file_path.is_file():
            logger.info(f"Serving static file: {file_path}")
            return FileResponse(file_path)
        
        # Try full path
        file_path = STATIC_DIR / clean_path
        if file_path.is_file():
            logger.info(f"Serving static file: {file_path}")
            return FileResponse(file_path)
    
    # Not a file request → serve index.html for SPA routing
    logger.info(f"Serving index.html for route: /{full_path}")
    return _index_file()







# from pathlib import Path
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from loguru import logger
# import os

# from app.config import settings
# from app.state import get_bundle
# from app.routers import query as query_router

# app = FastAPI(title="RAG Hybrid API", version="1.0.0")

# # ---------- CORS ----------
# origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------- Paths ----------
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# STATIC_DIR = PROJECT_ROOT / "static"
# DOCS_BASE_PATH = Path(os.getenv("BASE_DIR", r"C:\sms"))

# # ---------- Routers ----------
# app.include_router(query_router.router)
# app.include_router(query_router.router, prefix="/{client_id}")

# # ---------- Health Endpoint ----------
# @app.get("/health")
# async def health():
#     """Docker & uptime health check."""
#     return JSONResponse({"status": "ok", "source": "FastAPI", "frontend": "handled-by-nginx"})

# # ---------- Serve Tenant Documents ----------
# @app.get("/{client_id}/docs/{filename:path}")
# def serve_document(client_id: str, filename: str):
#     """
#     Serves HTML docs from BASE_DIR/<client_id>/documents/<filename>.
#     """
#     base = (DOCS_BASE_PATH / client_id / "documents").resolve()
#     full = (base / filename).resolve()

#     if not str(full).startswith(str(base)) or not full.is_file():
#         logger.error(f"Document not found: {full}")
#         raise HTTPException(status_code=404, detail="Document not found")

#     logger.info(f"Serving document: {full}")
#     return FileResponse(str(full))

# # ---------- SPA Route Handling (SAFE FALLBACK) ----------
# def _index_file() -> FileResponse | JSONResponse:
#     """Safe index.html fallback — only if it actually exists."""
#     index_path = STATIC_DIR / "index.html"
#     if index_path.is_file():
#         return FileResponse(index_path, media_type="text/html")
#     else:
#         # Return a JSON response instead of crashing
#         logger.warning(f"index.html not found at {index_path}, returning JSON fallback.")
#         return JSONResponse({"message": "Frontend is served separately via React/Nginx."})

# @app.get("/", response_class=HTMLResponse)
# def serve_root():
#     """Serve root or fallback JSON if no static frontend exists."""
#     return _index_file()

# @app.get("/{full_path:path}")
# async def serve_spa_catchall(full_path: str):
#     """
#     Handles static file routing or SPA fallback safely:
#     - If file exists → serve it
#     - Else → return JSON fallback (instead of crashing)
#     """
#     clean_path = full_path.rstrip('/')
#     filename = clean_path.split('/')[-1] if clean_path else ''

#     # 1️⃣ Try to serve a static file (if directory exists)
#     if filename and '.' in filename and STATIC_DIR.exists():
#         file_path = STATIC_DIR / clean_path
#         if file_path.is_file():
#             logger.info(f"Serving static file: {file_path}")
#             return FileResponse(file_path)

#     # 2️⃣ Fallback to index.html or JSON
#     logger.info(f"SPA route fallback triggered for: /{full_path}")
#     return _index_file()

# from pathlib import Path
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from loguru import logger
# import os
# from app.config import settings
# from app.routers import query as query_router

# app = FastAPI(title="RAG Hybrid API", version="1.0.0")

# # CORS
# origins = [o.strip() for o in settings.allow_origins.split(",")] if settings.allow_origins else ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Paths
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# STATIC_DIR = PROJECT_ROOT / "static"
# DOCS_BASE_PATH = Path(os.getenv("BASE_DIR", r"C:\sms"))

# # Mount static files (recommended for serving JS/CSS)
# app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# # Routers
# app.include_router(query_router.router)
# app.include_router(query_router.router, prefix="/{client_id}")

# # Health Endpoint
# @app.get("/health")
# async def health():
#     """Docker & uptime health check."""
#     return JSONResponse({"status": "ok", "source": "FastAPI"})

# # Serve tenant documents
# @app.get("/{client_id}/docs/{filename:path}")
# def serve_document(client_id: str, filename: str):
#     """Serves HTML docs from BASE_DIR/<client_id>/documents/<filename>."""
#     base = (DOCS_BASE_PATH / client_id / "documents").resolve()
#     full = (base / filename).resolve()
#     if not str(full).startswith(str(base)) or not full.is_file():
#         logger.error(f"Document not found: {full}")
#         raise HTTPException(status_code=404, detail="Document not found")
#     logger.info(f"Serving: {full}")
#     return FileResponse(str(full))

# # SPA routes
# def _index_file() -> FileResponse | JSONResponse:
#     """Safe index.html fallback."""
#     index_path = STATIC_DIR / "index.html"
#     if index_path.is_file():
#         return FileResponse(index_path, media_type="text/html")
#     else:
#         logger.warning(f"index.html not found at {index_path}, returning JSON fallback.")
#         return JSONResponse({"message": "Frontend not found. Please check if React build artifacts are correctly copied."})

# @app.get("/", response_class=HTMLResponse)
# def serve_root():
#     """Serve root or fallback JSON."""
#     return _index_file()

# @app.get("/{full_path:path}")
# async def serve_spa_catchall(full_path: str):
#     """
#     Smart static file server + SPA router:
#     - If file exists → serve it
#     - Else → serve index.html or JSON fallback
#     """
#     clean_path = full_path.rstrip('/')
#     filename = clean_path.split('/')[-1] if clean_path else ''
#     if filename and '.' in filename and STATIC_DIR.exists():
#         file_path = STATIC_DIR / clean_path
#         if file_path.is_file():
#             logger.info(f"Serving static file: {file_path}")
#             return FileResponse(file_path)
#     logger.info(f"Serving index.html for route: /{full_path}")
#     return _index_file()
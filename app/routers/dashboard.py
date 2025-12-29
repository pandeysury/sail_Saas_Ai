# app/routers/dashboard.py - RAG Analytics Dashboard API
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any
import sqlite3
import json
from datetime import datetime, timedelta
from app.auth import authenticate

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# IMPORTANT: Update this path to your query_logs.db location
QUERY_LOGS_DB = "/home/ubuntu/Desktop/Rag_System_Project/sail_Saas_Ai/query_logs.db"

@router.get("/test")
async def test_endpoint():
    """Test endpoint without authentication"""
    return {"status": "ok", "message": "Dashboard API is working"}

@router.get("/test-auth")
async def test_auth(user: str = Depends(authenticate)):
    """Test endpoint with authentication"""
    return {"status": "ok", "user": user, "message": "Authentication working"}

def get_db():
    """Get database connection"""
    try:
        conn = sqlite3.connect(QUERY_LOGS_DB)
        conn.row_factory = sqlite3.Row
        # Test connection
        conn.execute("SELECT 1").fetchone()
        return conn
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/overview")
async def get_overview(
    client_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """Get dashboard overview metrics - NO AUTH for testing"""
    try:
        conn = get_db()
        since_date = datetime.now() - timedelta(days=days)
        
        where_clause = "WHERE timestamp >= ?"
        params = [since_date.isoformat()]
        
        if client_id:
            where_clause += " AND user_org = ?"
            params.append(client_id)
        
        # Total queries
        total_queries = conn.execute(
            f"SELECT COUNT(*) FROM query_logs {where_clause}", params
        ).fetchone()[0]
        
        # Average quality (top reference score)
        avg_quality = conn.execute(
            f"SELECT AVG(COALESCE(top_reference_score, 0)) FROM query_logs {where_clause}", params
        ).fetchone()[0] or 0
        
        # Average response time
        avg_response_time = conn.execute(
            f"SELECT AVG(total_time_ms) FROM query_logs {where_clause}", params
        ).fetchone()[0] or 0
        
        # Active clients (all time, exclude None values)
        active_clients = conn.execute(
            "SELECT COUNT(DISTINCT user_org) FROM query_logs WHERE user_org IS NOT NULL"
        ).fetchone()[0]
        
        conn.close()
        
        return {
            "total_queries": total_queries,
            "avg_quality": round(avg_quality, 3),
            "avg_response_time": round(avg_response_time, 1),
            "active_clients": active_clients
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_queries": 0,
            "avg_quality": 0,
            "avg_response_time": 0,
            "active_clients": 0
        }

@router.get("/queries")
async def get_queries(
    client_id: Optional[str] = Query(None),
    query_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=365),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get paginated query logs - NO AUTH for testing"""
    try:
        conn = get_db()
        
        where_conditions = []
        params = []
        
        # Only add date filter if days is provided
        if days is not None:
            since_date = datetime.now() - timedelta(days=days)
            where_conditions.append("timestamp >= ?")
            params.append(since_date.isoformat())
        
        if client_id:
            where_conditions.append("user_org = ?")
            params.append(client_id)
            
        if query_type:
            where_conditions.append("query_intent = ?")
            params.append(query_type)
            
        if search:
            where_conditions.append("original_query LIKE ?")
            params.append(f"%{search}%")
        
        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM query_logs {where_clause}", params
        ).fetchone()[0]
        
        # Get queries
        queries = conn.execute(f"""
            SELECT id, original_query, query_intent, user_org, timestamp,
                   total_time_ms, top_reference_score, is_compound, confidence_score
            FROM query_logs {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()
        
        conn.close()
        
        return {
            "queries": [dict(q) for q in queries],
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        return {
            "error": str(e),
            "queries": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

@router.get("/queries/{query_id}")
async def get_query_detail(
    query_id: int,
    user: str = Depends(authenticate)
):
    """Get detailed query information"""
    conn = get_db()
    try:
        query = conn.execute(
            "SELECT * FROM query_logs WHERE id = ?", [query_id]
        ).fetchone()
        
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return dict(query)
    finally:
        conn.close()

@router.get("/performance/timeline")
async def get_performance_timeline(
    client_id: Optional[str] = Query(None),
    days: int = Query(7, ge=1, le=90),
    user: str = Depends(authenticate)
):
    """Get daily performance timeline"""
    conn = get_db()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        where_clause = "WHERE timestamp >= ?"
        params = [since_date.isoformat()]
        
        if client_id:
            where_clause += " AND user_org = ?"
            params.append(client_id)
        
        timeline = conn.execute(f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as query_count,
                AVG(total_time_ms) as avg_response_time,
                AVG(COALESCE(top_reference_score, 0)) as avg_quality
            FROM query_logs {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, params).fetchall()
        
        return [dict(row) for row in timeline]
    finally:
        conn.close()

@router.get("/clients")
async def get_client_stats(
    days: int = Query(30, ge=1, le=365),
    user: str = Depends(authenticate)
):
    """Get per-client statistics"""
    conn = get_db()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        clients = conn.execute("""
            SELECT 
                user_org as client_id,
                COUNT(*) as query_count,
                AVG(total_time_ms) as avg_response_time,
                AVG(COALESCE(top_reference_score, 0)) as avg_quality
            FROM query_logs 
            WHERE timestamp >= ?
            GROUP BY user_org
            ORDER BY query_count DESC
        """, [since_date.isoformat()]).fetchall()
        
        return [dict(row) for row in clients]
    finally:
        conn.close()

@router.get("/query-types")
async def get_query_types(
    client_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    user: str = Depends(authenticate)
):
    """Get query type distribution"""
    conn = get_db()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        where_clause = "WHERE timestamp >= ?"
        params = [since_date.isoformat()]
        
        if client_id:
            where_clause += " AND user_org = ?"
            params.append(client_id)
        
        types = conn.execute(f"""
            SELECT 
                COALESCE(query_intent, 'unknown') as query_type,
                COUNT(*) as count
            FROM query_logs {where_clause}
            GROUP BY query_intent
            ORDER BY count DESC
        """, params).fetchall()
        
        return [dict(row) for row in types]
    finally:
        conn.close()

@router.get("/export")
async def export_data(
    client_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    format: str = Query("json", regex="^(json|csv)$"),
    user: str = Depends(authenticate)
):
    """Export query data"""
    conn = get_db()
    try:
        since_date = datetime.now() - timedelta(days=days)
        
        where_clause = "WHERE timestamp >= ?"
        params = [since_date.isoformat()]
        
        if client_id:
            where_clause += " AND user_org = ?"
            params.append(client_id)
        
        queries = conn.execute(f"""
            SELECT * FROM query_logs {where_clause}
            ORDER BY timestamp DESC
        """, params).fetchall()
        
        data = [dict(row) for row in queries]
        
        if format == "csv":
            import csv
            import io
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return {"data": output.getvalue(), "format": "csv"}
        
        return {"data": data, "format": "json"}
    finally:
        conn.close()
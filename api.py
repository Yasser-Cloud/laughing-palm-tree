"""
FastAPI application - Search API and UI
"""

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import os

import config
from models import get_dense_model, get_sparse_model
from elastic_client import get_es_client
from watcher import get_watcher


# Initialize FastAPI
app = FastAPI(title="Arabic Document Search")

# Templates - use absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Request models
class SearchRequest(BaseModel):
    query: str
    size: int = 10
    use_bm25: bool = True
    use_dense: bool = True
    use_sparse: bool = True
    use_frr: bool = True


# Response models
class SearchResult(BaseModel):
    score: float
    document_id: str
    text: str
    metadata: dict


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - Search UI"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/search")
async def search_ui(request: Request, q: str = Query(""), size: int = Query(10),
                    use_bm25: bool = Query(True), use_dense: bool = Query(True), use_sparse: bool = Query(True), use_frr: bool = Query(True)):
    """Search page with results"""
    results = []
    if q:
        results = await _perform_search(q, size, use_bm25, use_dense, use_sparse, use_frr)
    return templates.TemplateResponse(request, "search.html", {
        "query": q,
        "results": results,
        "size": size,
        "use_bm25": use_bm25,
        "use_dense": use_dense,
        "use_sparse": use_sparse,
        "use_frr": use_frr
    })


@app.post("/api/search")
async def api_search(request: SearchRequest):
    """API endpoint for search"""
    results = await _perform_search(
        request.query, 
        request.size,
        request.use_bm25,
        request.use_dense,
        request.use_sparse,
        request.use_frr
    )
    return {"query": request.query, "results": results}


@app.get("/api/document/{doc_id}")
async def get_document(doc_id: str):
    """Get full document text by ID"""
    es = get_es_client()
    doc = es.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": doc_id,
        "text": doc.get("text", ""),
        "metadata": doc.get("metadata", {}),
        "indexed_at": doc.get("indexed_at", "")
    }


@app.get("/api/stats")
async def get_stats():
    """Get index statistics"""
    es = get_es_client()
    stats = es.get_stats()
    watcher = get_watcher()
    status = watcher.get_status()
    return {
        "elasticsearch": stats,
        "watcher": status
    }


@app.post("/api/reindex")
async def reindex():
    """Reindex all cached documents"""
    es = get_es_client()
    success, failed = es.reindex_from_cache()
    return {"success": success, "failed": failed}


@app.delete("/api/index")
async def delete_index():
    """Delete the entire index"""
    es = get_es_client()
    es.delete_index()
    return {"status": "deleted"}


async def _perform_search(query: str, size: int = 10, use_bm25: bool = True, use_dense: bool = True, use_sparse: bool = True, use_frr: bool = True):
    """Perform hybrid search"""
    import asyncio
    es = get_es_client()
    
    dense_vec = None
    sparse_vec = None
    
    if use_dense:
        dense_model = get_dense_model()
        dense_vec = dense_model.encode(query)[0]
    
    if use_sparse:
        sparse_model = get_sparse_model()
        sparse_vec = sparse_model.encode_query(query)[0]
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, es.search, query, dense_vec, sparse_vec, size, use_bm25, use_frr)


# Ngrok setup
def start_ngrok():
    """Start ngrok tunnel for public access"""
    if not config.USE_NGROK:
        return None
    
    try:
        from pyngrok import ngrok
        
        if config.NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
        
        public_url = ngrok.connect(config.API_PORT)
        return public_url
    except Exception as e:
        print(f"[Ngrok] Error: {e}")
        return None


def run_server():
    """Run the FastAPI server"""
    import uvicorn
    
    public_url = start_ngrok()
    
    print(f"\n{'='*60}")
    print(f"🚀 URL المحلي:  http://{config.API_HOST}:{config.API_PORT}")
    if public_url:
        print(f"🔗 العام:      {public_url}")
    else:
        print(f"⚠️  Ngrok غير مفعل - USE_NGROK=False في config.py للتفعيل")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="warning")


if __name__ == "__main__":
    run_server()

"""
main.py

FastAPI backend that exposes the semantic search engine as a simple
HTTP API for the React frontend to call.

Run with:
    uvicorn main:app --reload

Then visit http://localhost:8000/docs for interactive API docs
(FastAPI generates this automatically).
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import threading
from search import get_search_index

app = FastAPI(title="Semantic Search API")

# CORS: allows the React frontend (running on a different port/domain)
# to call this API from the browser. In production, replace "*" with
# your actual deployed frontend URL for better security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up_in_background():
    """
    Kick off model loading + index building on a background thread as soon
    as the server starts, instead of waiting for the first search request.
    This runs AFTER uvicorn has already opened its port, so it never
    delays startup or blocks a platform's health check -- it just means
    the first real search is likely to be fast instead of slow, since
    loading is probably already done (or in progress) by the time a user
    actually searches.
    """
    threading.Thread(target=get_search_index, daemon=True).start()


@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}


@app.get("/documents")
def get_documents():
    """Return all documents in the index (used to show 'browse all FAQs')."""
    return get_search_index().documents


@app.get("/search")
def search(
    q: str = Query(..., description="The user's search query"),
    top_k: int = Query(5, description="Number of results to return"),
    mode: str = Query("semantic", description="'semantic' or 'keyword'"),
):
    """
    Main search endpoint. Example:
        GET /search?q=how do I get my money back&mode=semantic
    """
    search_index = get_search_index()
    if mode == "keyword":
        results = search_index.keyword_search(q, top_k=top_k)
    else:
        results = search_index.semantic_search(q, top_k=top_k)

    return {
        "query": q,
        "mode": mode,
        "results": results,
    }


@app.get("/search/compare")
def compare(
    q: str = Query(..., description="The user's search query"),
    top_k: int = Query(5, description="Number of results to return"),
):
    """
    Returns BOTH semantic and keyword results for the same query side by
    side -- this powers the 'keyword vs semantic' demo view, which is the
    single most convincing feature of this project.
    """
    search_index = get_search_index()
    return {
        "query": q,
        "semantic_results": search_index.semantic_search(q, top_k=top_k),
        "keyword_results": search_index.keyword_search(q, top_k=top_k),
    }

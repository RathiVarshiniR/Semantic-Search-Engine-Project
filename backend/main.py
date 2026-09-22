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
from search import search_index

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


@app.get("/")
def root():
    return {"status": "ok", "message": "Semantic Search API is running"}


@app.get("/documents")
def get_documents():
    """Return all documents in the index (used to show 'browse all FAQs')."""
    return search_index.documents


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
    return {
        "query": q,
        "semantic_results": search_index.semantic_search(q, top_k=top_k),
        "keyword_results": search_index.keyword_search(q, top_k=top_k),
    }

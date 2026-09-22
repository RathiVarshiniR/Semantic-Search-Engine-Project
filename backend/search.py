"""
search.py

This file contains the core logic of the semantic search engine:
1. Load a small FAQ dataset
2. Convert each FAQ into an embedding vector (using a local, free model)
3. Given a user query, embed it and compare it against every stored
   embedding using cosine similarity
4. Return the top-k most similar results

For comparison/demo purposes, a simple keyword search is also included
so the frontend can show "keyword vs semantic" side by side.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")

# Common filler words that carry almost no meaning on their own. Without
# filtering these out, keyword search would "match" on words like "how",
# "do", "i", "is", making it look smarter than it actually is and hiding
# the real gap between keyword and semantic search.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "its", "our", "their", "this", "that", "these", "those",
    "how", "what", "when", "where", "why", "who", "which",
    "do", "does", "did", "doing", "done",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "have", "has", "had", "having",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under",
    "and", "or", "but", "if", "so", "as", "than", "then",
    "get", "got", "getting", "here", "there", "not", "no", "yes",
}

import string


def normalize_words(text: str) -> set:
    """Lowercase, strip punctuation, split into words, and drop stopwords."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    return {w for w in cleaned.split() if w not in STOPWORDS}

# ---------------------------------------------------------------------------
# 1. Load the embedding model (runs locally, free, no API key needed)
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is a small, fast, well-regarded sentence embedding model.
# The first time this runs it downloads the model (~80MB) and caches it
# locally; every run after that is instant and fully offline.
print("Loading embedding model... (first run may take a minute)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def load_documents():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Build the embedding index (done once, at startup)
# ---------------------------------------------------------------------------
class SearchIndex:
    def __init__(self):
        self.documents = load_documents()
        # We embed "question + answer" so search can match on either
        self.texts = [f"{d['question']} {d['answer']}" for d in self.documents]
        # model.encode turns a list of strings into a 2D numpy array
        # of shape (num_documents, embedding_dimension)
        self.embeddings = model.encode(self.texts, normalize_embeddings=True)

    def semantic_search(self, query: str, top_k: int = 5):
        """
        Embed the query, compare against every document embedding using
        cosine similarity, and return the top_k most similar documents.
        """
        query_embedding = model.encode([query], normalize_embeddings=True)[0]

        # Because embeddings are normalized (unit length), cosine similarity
        # simplifies to a plain dot product. This is the actual "search".
        scores = self.embeddings @ query_embedding  # shape: (num_documents,)

        # Get indices of the top_k highest scores, sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "question": doc["question"],
                "answer": doc["answer"],
                "score": float(scores[idx]),
            })
        return results

    def keyword_search(self, query: str, top_k: int = 5):
        """
        A simple keyword-overlap search, used only so the frontend can
        show "keyword search vs semantic search" side by side and make
        the value of embeddings visually obvious.
        """
        query_words = normalize_words(query)
        scored = []
        for doc in self.documents:
            doc_words = normalize_words(f"{doc['question']} {doc['answer']}")
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for overlap, doc in scored[:top_k]:
            results.append({
                "id": doc["id"],
                "question": doc["question"],
                "answer": doc["answer"],
                "score": overlap,
            })
        return results


# A single shared index instance, built once when the server starts
search_index = SearchIndex()


if __name__ == "__main__":
    # Quick manual test you can run directly: python search.py
    test_queries = [
        "how do I get my money back",
        "I forgot my login credentials",
        "is it safe to pay here",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        print("Semantic results:")
        for r in search_index.semantic_search(q, top_k=3):
            print(f"  [{r['score']:.3f}] {r['question']}")
        print("Keyword results:")
        kw = search_index.keyword_search(q, top_k=3)
        if not kw:
            print("  (no keyword matches found)")
        for r in kw:
            print(f"  [{r['score']}] {r['question']}")

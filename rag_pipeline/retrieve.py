# retrieve.py

from typing import List, Dict, Any, Optional

import google.generativeai as genai
from pinecone import Pinecone

from utils.config import (
    GEMINI_API_KEY,
    EMBED_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX,
)
from utils.logger import logger
from utils.retry import retry_with_backoff

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Lazy Pinecone connection - only connect when needed
_pc = None
_index = None

def get_pinecone_index():
    """Get Pinecone index, creating connection if needed."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(PINECONE_INDEX)
    return _index


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def _embed_query(query: str) -> List[float]:
    """Embed a query with retry logic."""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
    )
    return resp["embedding"]


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def _query_pinecone(query_kwargs: Dict[str, Any]) -> Any:
    """Query Pinecone with retry logic."""
    index = get_pinecone_index()
    return index.query(**query_kwargs)


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant chunks from Pinecone for a given query.

    Args:
        query: user question.
        top_k: number of chunks to retrieve.
        filters: optional metadata filter dict, e.g.
            {"doc_id": "<uuid>", "company": "PLTR", "year": 2024}

    Returns:
        List of dicts:
          {
            "text": str,
            "metadata": dict,
            "score": float
          }
    """
    logger.info(f"Retrieving top {top_k} chunks for query: {query[:100]}...")
    
    # Ensure minimum top_k for better results
    if top_k < 3:
        logger.warning(f"top_k={top_k} is too low, using minimum of 3")
        top_k = 3
    
    try:
        # 1) Embed query with Gemini
        q_emb = _embed_query(query)

        # 2) Build Pinecone query params
        # Retrieve more chunks than requested to filter by score later
        query_kwargs: Dict[str, Any] = {
            "vector": q_emb,
            "top_k": min(top_k * 2, 50),  # Get more candidates, filter by score
            "include_metadata": True,
        }
        if filters:
            query_kwargs["filter"] = filters
            logger.debug(f"Using filters: {filters}")

        # 3) Query Pinecone
        res = _query_pinecone(query_kwargs)

        # Support both SDK styles: res.matches or res["matches"]
        matches_raw = getattr(res, "matches", None)
        if matches_raw is None:
            matches_raw = res.get("matches", [])

        results: List[Dict[str, Any]] = []

        for m in matches_raw:
            # Handle object-style or dict-style
            if hasattr(m, "metadata"):
                meta = m.metadata
                score = m.score
            else:
                meta = m.get("metadata", {})
                score = m.get("score")

            # Filter out very low relevance scores (below 0.3)
            # This helps ensure we get meaningful chunks
            if score and score < 0.3:
                logger.debug(f"Skipping chunk with low score: {score:.3f}")
                continue

            results.append(
                {
                    "text": meta.get("text", ""),
                    "metadata": meta,
                    "score": score,
                    # expose helpful fields at top-level for UI convenience
                    "rows": meta.get("rows"),
                    "chunk_type": meta.get("chunk_type"),
                    "page": meta.get("page"),
                    "source_file": meta.get("source_file"),
                    "snippet": (meta.get("text", "")[:1000] if meta.get("text") else ""),
                }
            )

        # Take top_k results after filtering
        results = results[:top_k]
        
        logger.info(f"Retrieved {len(results)} chunks (after filtering)")
        if results and results[0].get("score"):
            logger.debug(f"Top chunk score: {results[0]['score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise

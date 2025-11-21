# retrieve.py

from typing import Dict, Any, List, Optional
import numpy as np
import google.generativeai as genai
import faiss

from utils.config import GEMINI_API_KEY, EMBED_MODEL
from utils.logger import logger
from utils.retry import retry_with_backoff
from rag_pipeline.embed_store import get_faiss_index, get_faiss_metadata

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def _embed_query(query: str) -> List[float]:
    """Embed a query string using Gemini."""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
    )
    return resp["embedding"]


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant chunks from FAISS for a given query.

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
            "score": float,
            "rows": list (if table),
            "chunk_type": str,
            "page": int,
            "source_file": str,
            "snippet": str
          }
    """
    logger.info(f"Retrieving top {top_k} chunks for query: {query[:100]}...")
    
    # Ensure minimum top_k for better results
    if top_k < 3:
        logger.warning(f"top_k={top_k} is too low, using minimum of 3")
        top_k = 3
    
    # Get FAISS index and metadata
    index = get_faiss_index()
    metadata_list = get_faiss_metadata()
    
    if index.ntotal == 0:
        logger.warning("FAISS index is empty")
        return []
    
    try:
        # 1) Embed query with Gemini
        q_emb = _embed_query(query)
        
        # 2) Convert query to numpy array
        query_vector = np.array([q_emb], dtype=np.float32)
        
        # 3) If filters are provided, we need to search more results and filter
        # then return top_k from filtered results
        search_k = top_k * 10 if filters else top_k * 2  # Search more if filtering
        
        # 4) Search in FAISS
        distances, indices = index.search(query_vector, min(search_k, index.ntotal))
        
        # 5) Get metadata for retrieved indices and apply filters
        results: List[Dict[str, Any]] = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
            
            if idx >= len(metadata_list):
                logger.warning(f"Index {idx} out of range for metadata list")
                continue
            
            meta = metadata_list[idx].copy()
            
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if meta.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Convert L2 distance to similarity score (lower distance = higher similarity)
            # Using inverse distance as score (with a small epsilon to avoid division by zero)
            score = 1.0 / (1.0 + dist)
            
            # Filter out very low relevance scores (below 0.3)
            if score < 0.3:
                logger.debug(f"Skipping chunk with low score: {score:.3f}")
                continue
            
            results.append({
                "text": meta.get("text", ""),
                "metadata": {k: v for k, v in meta.items() if k != "text"},
                "score": float(score),
                # expose helpful fields at top-level for UI convenience
                "rows": meta.get("rows"),
                "chunk_type": meta.get("chunk_type"),
                "page": meta.get("page"),
                "source_file": meta.get("source_file"),
                "snippet": (meta.get("text", "")[:1000] if meta.get("text") else ""),
            })
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        logger.info(f"Retrieved {len(results)} chunks (after filtering)")
        if results and results[0].get("score"):
            logger.debug(f"Top chunk score: {results[0]['score']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise

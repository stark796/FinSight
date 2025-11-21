# embed_store.py

import uuid
import pickle
import os
from typing import List, Dict, Any
from pathlib import Path

import faiss
import numpy as np
import google.generativeai as genai

from utils.config import (
    GEMINI_API_KEY,
    EMBED_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_DIM,
)
from utils.logger import logger
from utils.retry import retry_with_backoff

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Global FAISS index and metadata
_index = None
_metadata = []  # List of dicts, one per vector: {"id": str, "text": str, **metadata}


def _ensure_data_dir():
    """Ensure data directory exists."""
    data_dir = Path(FAISS_INDEX_PATH).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _load_faiss_index():
    """Load FAISS index and metadata from disk if they exist."""
    global _index, _metadata
    
    if _index is not None:
        return _index
    
    index_path = Path(FAISS_INDEX_PATH)
    metadata_path = Path(FAISS_METADATA_PATH)
    
    if index_path.exists() and metadata_path.exists():
        try:
            logger.info(f"Loading FAISS index from {index_path}")
            _index = faiss.read_index(str(index_path))
            
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, "rb") as f:
                _metadata = pickle.load(f)
            
            logger.info(f"Loaded {len(_metadata)} vectors from FAISS index")
        except Exception as e:
            logger.warning(f"Error loading existing index: {e}. Creating new index.")
            _index = None
            _metadata = []
    
    if _index is None:
        # Create new index: L2 distance, dimension from config
        logger.info(f"Creating new FAISS index with dimension {EMBEDDING_DIM}")
        _index = faiss.IndexFlatL2(EMBEDDING_DIM)
        _metadata = []
    
    return _index


def _save_faiss_index():
    """Save FAISS index and metadata to disk."""
    global _index, _metadata
    
    if _index is None:
        return
    
    _ensure_data_dir()
    index_path = Path(FAISS_INDEX_PATH)
    metadata_path = Path(FAISS_METADATA_PATH)
    
    try:
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(_index, str(index_path))
        
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "wb") as f:
            pickle.dump(_metadata, f)
        
        logger.info(f"Saved {len(_metadata)} vectors to FAISS index")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}", exc_info=True)
        raise


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def _embed_single(text: str) -> List[float]:
    """Embed a single text with retry logic."""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
    )
    return resp["embedding"]


def embed_chunks(chunks: List[Dict[str, Any]], upload_id: str | None = None) -> List[List[float]]:
    """
    Embed a list of chunks using Gemini.

    Each chunk is expected to be:
      {
        "text": str,
        "metadata": dict   # optional
      }

    Args:
        chunks: List of chunks to embed
        upload_id: Optional upload ID for progress tracking

    Returns:
        List of embedding vectors (same order as input).
    """
    def _embedding_text_for_chunk(c: Dict[str, Any]) -> str:
        # For table chunks, embed a concise summary (header + first few rows)
        meta = c.get("metadata", {}) or {}
        if meta.get("chunk_type") == "table" and meta.get("rows"):
            rows = meta.get("rows")
            try:
                # rows expected as list of lists; first row header
                header = [str(h) if h is not None else "" for h in rows[0]]
                max_rows = 3
                data_rows = rows[1:1+max_rows]
                lines = [" | ".join(header), "|" + "|".join(["---"]*len(header)) + "|"]
                for r in data_rows:
                    lines.append(" | ".join([str(cell) if cell is not None else "" for cell in r]))
                summary = "\n".join(lines)
                return f"Table (summary):\n{summary}"
            except Exception:
                # Fallback to plain text if rows malformed
                return c.get("text", "")
        # Default: use full text
        return c.get("text", "")

    texts = [_embedding_text_for_chunk(c) for c in chunks]
    vectors: List[List[float]] = []
    total = len(texts)

    logger.info(f"Embedding {total} chunks...")
    
    # Report initial progress
    if upload_id:
        from utils.progress import set_progress
        set_progress(upload_id, total, 0, "embedding")
    
    for i, text in enumerate(texts):
        try:
            vector = _embed_single(text)
            vectors.append(vector)
            
            # Report progress every chunk (or every 5 chunks for less overhead)
            if upload_id and (i + 1) % 5 == 0 or i == 0 or i == total - 1:
                set_progress(upload_id, total, i + 1, "embedding")
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{total} chunks")
        except Exception as e:
            logger.error(f"Error embedding chunk {i + 1}: {e}")
            raise

    logger.info(f"Successfully embedded {len(vectors)} chunks")
    
    # Report completion
    if upload_id:
        set_progress(upload_id, total, total, "storing")
    
    return vectors


def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]], upload_id: str | None = None) -> None:
    """
    Store chunks and their embeddings into FAISS.
    
    We store:
      - vectors in FAISS index
      - metadata in a list (one dict per vector): {"id": str, "text": str, **metadata}
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    if not chunks:
        logger.warning("No chunks to store in FAISS.")
        return

    # Load existing index
    index = _load_faiss_index()
    global _metadata

    # Convert embeddings to numpy array
    vectors_array = np.array(embeddings, dtype=np.float32)
    
    # Add vectors to index
    logger.info(f"Adding {len(vectors_array)} vectors to FAISS index...")
    index.add(vectors_array)
    
    # Store metadata for each chunk
    for chunk, vector in zip(chunks, embeddings):
        meta = {"text": chunk["text"], **chunk.get("metadata", {})}
        item_id = str(uuid.uuid4())
        meta["id"] = item_id
        _metadata.append(meta)
    
    # Save index and metadata to disk
    _save_faiss_index()
    
    # Report completion
    if upload_id:
        from utils.progress import set_progress, clear_progress
        set_progress(upload_id, len(chunks), len(chunks), "completed")
        # Clear after a delay (let frontend get final update)
        import threading
        def clear_after_delay():
            import time
            time.sleep(5)
            clear_progress(upload_id)
        threading.Thread(target=clear_after_delay, daemon=True).start()
    
    logger.info(f"Successfully stored {len(chunks)} chunks in FAISS index (total: {index.ntotal} vectors)")


def get_faiss_index():
    """Get FAISS index, loading from disk if needed."""
    return _load_faiss_index()


def get_faiss_metadata():
    """Get FAISS metadata list."""
    _load_faiss_index()  # Ensure index is loaded
    return _metadata


def delete_vectors_by_doc_id(doc_id: str) -> None:
    """
    Delete all vectors with metadata doc_id == doc_id.
    This rebuilds the index without those vectors.
    """
    global _index, _metadata
    
    _load_faiss_index()
    
    if not _metadata:
        logger.info("No vectors to delete")
        return
    
    # Find indices to keep (those NOT matching doc_id)
    indices_to_keep = []
    for i, meta in enumerate(_metadata):
        if meta.get("doc_id") != doc_id:
            indices_to_keep.append(i)
    
    if len(indices_to_keep) == len(_metadata):
        logger.info(f"No vectors found with doc_id={doc_id}")
        return
    
    logger.info(f"Deleting {len(_metadata) - len(indices_to_keep)} vectors with doc_id={doc_id}")
    
    # Rebuild index and metadata with only kept vectors
    old_index = _index
    old_metadata = _metadata
    
    # Create new index
    _index = faiss.IndexFlatL2(EMBEDDING_DIM)
    _metadata = []
    
    # Re-add only the vectors we want to keep
    for i in indices_to_keep:
        # Get the vector from old index
        vector = old_index.reconstruct(i)
        vector = vector.reshape(1, -1).astype(np.float32)
        
        # Add to new index
        _index.add(vector)
        
        # Add metadata
        _metadata.append(old_metadata[i].copy())
    
    # Save updated index
    _save_faiss_index()
    
    logger.info(f"Deleted vectors for doc_id={doc_id}. Remaining vectors: {len(_metadata)}")

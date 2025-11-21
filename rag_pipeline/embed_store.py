# embed_store.py

import uuid
from typing import List, Dict, Any

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
def _embed_single(text: str) -> List[float]:
    """Embed a single text with retry logic."""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
    )
    return resp["embedding"]


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Embed a list of chunks using Gemini.

    Each chunk is expected to be:
      {
        "text": str,
        "metadata": dict   # optional
      }

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

    logger.info(f"Embedding {len(texts)} chunks...")
    for i, text in enumerate(texts):
        try:
            vector = _embed_single(text)
            vectors.append(vector)
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error embedding chunk {i + 1}: {e}")
            raise

    logger.info(f"Successfully embedded {len(vectors)} chunks")
    return vectors


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def store_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """
    Store chunks and their embeddings into Pinecone.
    
    Batches upserts to stay under Pinecone's 2MB request size limit.

    We store:
      - id: random UUID
      - values: embedding vector
      - metadata: { "text": ..., <user metadata> }
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    items = []
    for chunk, vector in zip(chunks, embeddings):
        meta = {"text": chunk["text"], **chunk.get("metadata", {})}
        item_id = str(uuid.uuid4())

        # Pinecone expects (id, vector, metadata)
        items.append((item_id, vector, meta))

    if not items:
        logger.warning("No items to upsert into Pinecone.")
        return

    # Pinecone has a 2MB request size limit, so we batch the upserts
    # Each item is roughly: vector (768 dims * 4 bytes = ~3KB) + metadata (~1-5KB) = ~5-10KB
    # To stay under 2MB, we use batches of 100 items (~500KB-1MB per batch)
    BATCH_SIZE = 100
    total_items = len(items)
    
    logger.info(f"Upserting {total_items} vectors into Pinecone in batches of {BATCH_SIZE}...")
    index = get_pinecone_index()
    
    for i in range(0, total_items, BATCH_SIZE):
        batch = items[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.debug(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} items)...")
        index.upsert(batch)
        logger.debug(f"Completed batch {batch_num}/{total_batches}")
    
    logger.info(f"Successfully upserted {total_items} vectors into Pinecone in {total_batches} batch(es).")

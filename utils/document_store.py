# utils/document_store.py

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock

from utils.logger import logger

# Simple file-based document metadata store
# In production, use a proper database (PostgreSQL, MongoDB, etc.)
DOC_STORE_FILE = Path("data") / "documents.json"
DOC_STORE_FILE.parent.mkdir(exist_ok=True)

_store_lock = Lock()
_documents: Dict[str, Dict] = {}


def _load_documents() -> Dict[str, Dict]:
    """Load documents from file."""
    global _documents
    if DOC_STORE_FILE.exists():
        try:
            with open(DOC_STORE_FILE, "r") as f:
                _documents = json.load(f)
        except Exception as e:
            logger.error(f"Error loading document store: {e}")
            _documents = {}
    return _documents


def _save_documents() -> None:
    """Save documents to file."""
    try:
        with open(DOC_STORE_FILE, "w") as f:
            json.dump(_documents, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving document store: {e}")


def register_document(
    doc_id: str,
    filename: str,
    file_path: str,
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> None:
    """Register a new document in the store."""
    with _store_lock:
        _load_documents()
        _documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_path": file_path,
            "company": company,
            "year": year,
            "uploaded_at": datetime.utcnow().isoformat(),
            "chunk_count": 0,  # Will be updated after ingestion
            "indexing_status": "indexing",  # "indexing", "completed", "failed"
        }
        _save_documents()
        logger.info(f"Registered document: {doc_id} ({filename})")


def get_document(doc_id: str) -> Optional[Dict]:
    """Get document metadata by doc_id."""
    with _store_lock:
        _load_documents()
        return _documents.get(doc_id)


def list_documents() -> List[Dict]:
    """List all registered documents."""
    with _store_lock:
        _load_documents()
        return list(_documents.values())


def delete_document(doc_id: str) -> bool:
    """Delete document metadata. Returns True if found and deleted."""
    with _store_lock:
        _load_documents()
        if doc_id in _documents:
            doc = _documents.pop(doc_id)
            _save_documents()
            logger.info(f"Deleted document metadata: {doc_id}")
            
            # Optionally delete the file
            file_path = doc.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
            
            return True
        return False


def update_chunk_count(doc_id: str, count: int) -> None:
    """Update the chunk count for a document."""
    with _store_lock:
        _load_documents()
        if doc_id in _documents:
            _documents[doc_id]["chunk_count"] = count
            _save_documents()


def set_indexing_status(doc_id: str, status: str) -> None:
    """Set the indexing status for a document. Status: 'indexing', 'completed', 'failed'."""
    with _store_lock:
        _load_documents()
        if doc_id in _documents:
            _documents[doc_id]["indexing_status"] = status
            _save_documents()
            logger.info(f"Document {doc_id} indexing status: {status}")


def is_document_indexed(doc_id: str) -> bool:
    """Check if a document is fully indexed and ready for queries."""
    with _store_lock:
        _load_documents()
        doc = _documents.get(doc_id)
        if not doc:
            return False
        status = doc.get("indexing_status")
        # For backward compatibility: if status doesn't exist, assume completed
        if status is None:
            return True
        return status == "completed"


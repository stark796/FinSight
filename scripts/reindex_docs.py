#!/usr/bin/env python3
"""
scripts/reindex_docs.py

Usage:
  # Re-ingest all documents (will delete existing vectors per doc before ingest)
  python scripts/reindex_docs.py --all

  # Re-ingest specific documents by doc_id
  python scripts/reindex_docs.py <doc_id1> <doc_id2>

Notes:
  - This script will attempt to delete vectors from the Pinecone index for each
    document by filtering on `doc_id` and then re-run `ingest_pdf` using the
    stored `file_path` in your document store. It preserves the same `doc_id` so
    downstream references remain stable.
  - Deleting vectors is irreversible in the index; be careful when running.
"""
import argparse
import sys
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../FinSight/scripts
FIN_SIGHT_DIR = os.path.dirname(SCRIPT_DIR)  # .../FinSight
if FIN_SIGHT_DIR not in sys.path:
    # Ensure FinSight directory is on sys.path so `import utils` works
    sys.path.insert(0, FIN_SIGHT_DIR)
from utils.logger import logger
from utils.document_store import list_documents, get_document
from rag_pipeline.ingest import ingest_pdf
from rag_pipeline.embed_store import get_pinecone_index


def delete_vectors_for_doc(doc_id: str) -> None:
    """Delete vectors in Pinecone that have metadata doc_id == doc_id."""
    try:
        index = get_pinecone_index()
        # Pinecone SDK supports delete by metadata filter
        index.delete(filter={"doc_id": doc_id})
        logger.info(f"Deleted vectors for doc_id={doc_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vectors for {doc_id}: {e}")


def reingest_doc(doc_id: str) -> None:
    """Delete vectors for doc_id and re-run ingestion using stored file_path."""
    doc = get_document(doc_id)
    if not doc:
        logger.error(f"Document not found in store: {doc_id}")
        return

    file_path = doc.get("file_path") or doc.get("file_path_saved")
    if not file_path:
        logger.error(f"No file_path available for doc {doc_id}; skipping")
        return

    logger.info(f"Re-indexing doc_id={doc_id} file={file_path}")

    # Delete existing vectors for this doc to avoid duplicates
    delete_vectors_for_doc(doc_id)

    # Small pause to ensure deletion propagates (Pinecone may be eventual-consistent)
    time.sleep(1)

    # Re-ingest using same doc_id
    try:
        ingest_pdf(file_path, doc_id=doc_id, company=doc.get("company"), year=doc.get("year"))
        logger.info(f"Re-ingested {doc_id} successfully")
    except Exception as e:
        logger.error(f"Failed to re-ingest {doc_id}: {e}")


def reingest_all() -> None:
    docs = list_documents()
    logger.info(f"Found {len(docs)} documents to re-ingest")
    for d in docs:
        reingest_doc(d["doc_id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-index documents: delete vectors by doc_id and re-ingest files.")
    parser.add_argument("doc_ids", nargs="*", help="Document IDs to re-ingest (omit if using --all)")
    parser.add_argument("--all", action="store_true", help="Re-ingest all documents from the document store")

    args = parser.parse_args(argv)

    if args.all:
        reingest_all()
        return

    if not args.doc_ids:
        parser.print_help()
        return

    for did in args.doc_ids:
        reingest_doc(did)


if __name__ == "__main__":
    main()

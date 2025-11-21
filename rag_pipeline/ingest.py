# ingest.py

import os
import uuid
import re
import pdfplumber
from typing import List, Dict, Any

from .embed_store import embed_chunks, store_chunks
from utils.logger import logger
from utils.config import CHUNK_SIZE


def chunk_page_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = 200) -> List[str]:
    """
    Improved chunking: split text at sentence boundaries when possible.
    Falls back to paragraph boundaries, then fixed-size chunks.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks for context preservation
    
    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    # Try to split by sentences first (period, exclamation, question mark followed by space)
    sentences = re.split(r'([.!?]\s+)', text)
    # Rejoin sentences with their punctuation
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2) if sentences[i].strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed max_chars, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Fallback: if no good chunks created, use fixed-size splitting
    if not chunks:
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end].strip())
            start = end - overlap  # Overlap for context
    
    return [chunk for chunk in chunks if chunk]


def format_table_as_text(table: List[List]) -> str:
    """
    Convert a table (list of lists) into a readable text format.
    Uses markdown-style table format for better readability.
    """
    if not table or not table[0]:
        return ""
    
    # Format as markdown table
    lines = []
    
    # Header row
    if len(table) > 0:
        header = " | ".join(str(cell) if cell is not None else "" for cell in table[0])
        lines.append(f"| {header} |")
        lines.append("|" + "|".join(["---"] * len(table[0])) + "|")
    
    # Data rows
    for row in table[1:]:
        if row:
            row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
            lines.append(f"| {row_text} |")
    
    return "\n".join(lines)


def ingest_pdf(
    file_path: str,
    doc_id: str | None = None,
    company: str | None = None,
    year: int | None = None,
) -> str:
    """
    Parse PDF → chunk → embed with Gemini → store in Pinecone.

    Args:
        file_path: Path to PDF file
        doc_id: Optional document ID. If not provided, generates a new UUID.
        company: Optional company name
        year: Optional year

    Returns:
        doc_id (str) for later retrieval during /ask.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    logger.info(f"Starting ingestion for {file_path} (doc_id={doc_id})")

    all_chunks: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            num_pages = len(pdf.pages)
            logger.info(f"PDF has {num_pages} pages")

            for i, page in enumerate(pdf.pages):
                page_number = i + 1
                try:
                    # Extract regular text
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    tables = page.extract_tables()
                    table_count = len(tables) if tables else 0
                    
                    if not text.strip() and table_count == 0:
                        logger.debug(f"Page {page_number} has no extractable text or tables")
                        continue

                    # Process regular text chunks
                    if text.strip():
                        page_chunks = chunk_page_text(text)
                        logger.debug(f"Page {page_number}: created {len(page_chunks)} text chunks")
                        
                        for idx, chunk_text in enumerate(page_chunks):
                            all_chunks.append(
                                {
                                    "text": chunk_text,
                                    "metadata": {
                                        "doc_id": doc_id,
                                        "company": company,
                                        "year": year,
                                        "page": page_number,
                                        "chunk_index": idx,
                                        "chunk_type": "text",
                                        "source_file": os.path.basename(file_path),
                                    },
                                }
                            )
                    
                    # Process tables
                    if tables:
                        logger.debug(f"Page {page_number}: found {table_count} table(s)")
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 0:
                                table_text = format_table_as_text(table)
                                if table_text.strip():
                                    all_chunks.append(
                                        {
                                            "text": f"Table {table_idx + 1} from page {page_number}:\n{table_text}",
                                            "metadata": {
                                                "doc_id": doc_id,
                                                "company": company,
                                                "year": year,
                                                "page": page_number,
                                                "chunk_index": table_idx,
                                                "chunk_type": "table",
                                                "table_index": table_idx,
                                                "rows": table,
                                                "source_file": os.path.basename(file_path),
                                            },
                                        }
                                    )
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_number}: {e}")
                    continue

        if not all_chunks:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return doc_id

        logger.info(f"Extracted {len(all_chunks)} text chunks from PDF")

        # Embed and store into Pinecone
        embeddings = embed_chunks(all_chunks)
        store_chunks(all_chunks, embeddings)
        
        # Update document store with chunk count
        from utils.document_store import update_chunk_count
        update_chunk_count(doc_id, len(all_chunks))

        logger.info(f"Completed ingestion for doc_id={doc_id}")
        return doc_id
        
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {e}", exc_info=True)
        raise

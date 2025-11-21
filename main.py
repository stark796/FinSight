# main.py

import os
import shutil
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from rag_pipeline.retrieve import retrieve
from rag_pipeline.generate import generate_answer
from rag_pipeline.ingest import ingest_pdf
from rag_pipeline.embed_store import delete_vectors_by_doc_id
from utils.logger import logger
from utils.document_store import (
    register_document,
    get_document,
    list_documents,
    delete_document,
    set_indexing_status,
    is_document_indexed,
)

# -----------------------------
# FastAPI app & CORS
# -----------------------------
app = FastAPI(
    title="FinSight - RAG Analyst Backend",
    description="Upload PDFs and ask questions using Gemini + FAISS",
    version="2.0.0",
)

# CORS Configuration
# TODO: In production, replace "*" with your frontend URL(s)
# Example: allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # SECURITY: Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf"}

# -----------------------------
# Models
# -----------------------------
class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    company: str | None = None
    year: int | None = None
    message: str
    file_size: int | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=20)  # Increased default for better results
    # Either query a single document (`doc_id`) or multiple documents (`doc_ids`).
    doc_id: str | None = Field(None, description="Document ID to query")
    doc_ids: List[str] | None = Field(None, description="List of document IDs to query across multiple documents")
    company: str | None = None
    companies: List[str] | None = None
    year: int | None = None
    conversation_history: List[Dict[str, str]] | None = Field(None, description="Previous conversation messages for context")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_has_doc_or_company(self):
        """Ensure at least one of doc_id/doc_ids or company/companies is provided."""
        has_doc = bool(self.doc_id or self.doc_ids)
        has_company = bool(self.company or self.companies)
        if not (has_doc or has_company):
            raise ValueError("Either `doc_id`/`doc_ids` or `company`/`companies` must be provided")
        return self


class Source(BaseModel):
    page: int | None = None
    section: str | None = None
    score: float | None = None
    snippet: str | None = None
    doc_id: str | None = None


class VerificationItem(BaseModel):
    claim_token: str
    claim_value: float | None = None
    matched: bool
    best_source_index: int | None = None
    source_token: str | None = None
    source_value: float | None = None
    diff: float | None = None
    rel_error: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    verification: List[VerificationItem] | None = None
    fact_check: List[Dict[str, Any]] | None = None
    company_results: List[Dict[str, Any]] | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    company: str | None = None
    year: int | None = None
    uploaded_at: str
    chunk_count: int
    file_path: str | None = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str
    deleted: bool


# -----------------------------
# Helper Functions
# -----------------------------
def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """Validate uploaded file."""
    if not file.filename:
        return False, "Filename is required"
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported"
    
    return True, ""


async def get_file_size(file: UploadFile) -> int:
    """Get file size from upload."""
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    return size


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to avoid conflicts."""
    ext = Path(original_filename).suffix
    unique_id = str(uuid.uuid4())[:8]
    base_name = Path(original_filename).stem
    return f"{base_name}_{unique_id}{ext}"


# -----------------------------
# Endpoints
# -----------------------------

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    company: str | None = Form(None),
    year: int | None = Form(None),
):
    """
    Upload and index a PDF document.
    
    Process:
      1) Validate file (type, size)
      2) Save the file with unique name
      3) Parse + chunk + embed + store in FAISS
      4) Register document in metadata store
      5) Return doc_id for later queries
    """
    doc_id = ""
    save_path = None
    
    try:
        # Validate file
        is_valid, error_msg = validate_file(file)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        
        # Check file size
        file_size = await get_file_size(file)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE / 1024 / 1024} MB)",
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty",
            )
        
        # Generate unique filename and save
        unique_filename = generate_unique_filename(file.filename)
        save_path = UPLOAD_DIR / unique_filename
        
        logger.info(f"Uploading file: {file.filename} ({file_size / 1024:.2f} KB)")
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Generate doc_id and upload_id before ingestion
        doc_id = str(uuid.uuid4())
        upload_id = f"upload-{doc_id}"
        
        # Initialize progress tracking
        from utils.progress import set_progress
        set_progress(upload_id, 100, 0, "parsing")  # Initial state
        
        # Register document before ingestion (status: indexing)
        register_document(
            doc_id=doc_id,
            filename=file.filename,
            file_path=str(save_path),
            company=company,
            year=year,
        )
        set_indexing_status(doc_id, "indexing")
        
        # Run ingestion synchronously in executor to allow progress tracking
        # but wait for completion before returning
        logger.info(f"Starting ingestion for doc_id={doc_id}")
        try:
            # Run in thread pool to avoid blocking the event loop
            # This allows progress tracking while still waiting for completion
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                ingest_pdf,
                str(save_path),
                doc_id,
                company,
                year,
                upload_id,
            )
            
            # ingest_pdf will set status to "completed" or "failed"
            logger.info(f"Successfully uploaded and indexed: {doc_id}")
            
            return UploadResponse(
                doc_id=doc_id,
                filename=file.filename,
                company=company,
                year=year,
                message="File uploaded and indexed successfully.",
                file_size=file_size,
                upload_id=upload_id,
            )
        except Exception as ingest_error:
            # ingest_pdf will have already set status to "failed" in its exception handler
            logger.error(f"Error during ingestion: {ingest_error}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error indexing document: {str(ingest_error)}",
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in /upload: {e}", exc_info=True)
        
        # Clean up file if it was saved
        if save_path and save_path.exists():
            try:
                save_path.unlink()
                logger.info(f"Cleaned up file: {save_path}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up file: {cleanup_error}")
        
        # Remove from document store if registered
        if doc_id:
            try:
                delete_document(doc_id)
            except Exception:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error while uploading/indexing the file. Please try again.",
        )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question about a specific uploaded document.
    
    Process:
      1) Validate document exists
      2) Retrieve relevant chunks with filters (doc_id + optional company/year)
      3) Generate an answer with Gemini
      4) Return answer + source snippets
    """
    try:
        # Resolve query scope: document-based, company-based, or company-comparison
        # 1) Document(s) provided (doc_id / doc_ids)
        # 2) Single company provided (company)
        # 3) Comparison across multiple companies (companies with length >= 2)

        # Helper to build sources list from context chunks
        def build_sources_from_ctx(ctx_list: List[Dict[str, Any]]) -> List[Source]:
            s: List[Source] = []
            for c in ctx_list:
                meta = c.get("metadata", {})
                snippet = c.get("text", "")
                s.append(
                    Source(
                        page=meta.get("page"),
                        section=meta.get("section"),
                        score=c.get("score"),
                        snippet=(snippet[:300] + ("..." if len(snippet) > 300 else "")),
                        doc_id=meta.get("doc_id"),
                    )
                )
            return s

        # If document IDs provided, prefer them
        if req.doc_ids or req.doc_id:
            doc_ids = req.doc_ids if req.doc_ids else [req.doc_id]
            valid_docs = []
            not_indexed_docs = []
            for did in doc_ids:
                d = get_document(did)
                if d:
                    if is_document_indexed(did):
                        valid_docs.append(did)
                    else:
                        not_indexed_docs.append(did)

            if not_indexed_docs:
                doc_names = [get_document(did).get("filename", did) for did in not_indexed_docs if get_document(did)]
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail=f"Document(s) still being indexed: {', '.join(doc_names)}. Please wait for indexing to complete.",
                )

            if not valid_docs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No valid documents found for provided doc_id(s)",
                )

            logger.info(f"Processing question for doc_ids={valid_docs}: {req.question[:100]}")

            aggregated_ctx: List[Dict[str, Any]] = []
            for did in valid_docs:
                # Use doc_id filter; also allow optional year/company narrowing
                filters: Dict[str, Any] = {"doc_id": did}
                if req.company:
                    filters["company"] = req.company
                if req.year:
                    filters["year"] = req.year

                ctx = retrieve(req.question, top_k=req.top_k, filters=filters)
                if ctx:
                    for c in ctx:
                        meta = c.get("metadata", {})
                        meta.setdefault("doc_id", did)
                        c["metadata"] = meta
                    aggregated_ctx.extend(ctx)

            if not aggregated_ctx:
                logger.warning(f"No context found for query: {req.question[:100]}")
                return AskResponse(
                    answer="No relevant information found in the provided documents.",
                    sources=[],
                    verification=[],
                )

            aggregated_ctx.sort(key=lambda x: (x.get("score") or 0), reverse=True)
            aggregated_ctx = aggregated_ctx[: req.top_k]

            gen_result = generate_answer(req.question, aggregated_ctx, req.conversation_history)
            answer_text = gen_result.get("answer") if isinstance(gen_result, dict) else str(gen_result)
            verification = gen_result.get("verification") if isinstance(gen_result, dict) else []
            sources = build_sources_from_ctx(aggregated_ctx)

            logger.info(f"Successfully generated answer for doc_ids={valid_docs}")
            return AskResponse(answer=answer_text, sources=sources, verification=verification)

        # Company comparison mode: multiple companies provided
        if req.companies and len(req.companies) >= 2:
            logger.info(f"Processing comparison for companies={req.companies}: {req.question[:100]}")
            company_results: List[Dict[str, Any]] = []
            # Keep raw contexts for later combined summary
            company_ctx_map: Dict[str, List[Dict[str, Any]]] = {}

            for comp in req.companies:
                # Retrieve top chunks for this company
                filters: Dict[str, Any] = {"company": comp}
                if req.year:
                    filters["year"] = req.year

                ctx = retrieve(req.question, top_k=req.top_k, filters=filters)
                if not ctx:
                    # Return an empty result set for this company but continue
                    company_results.append({"company": comp, "answer": "No documents found for this company.", "sources": []})
                    company_ctx_map[comp] = []
                    continue

                # annotate provenance
                for c in ctx:
                    meta = c.get("metadata", {})
                    if not meta.get("company"):
                        meta.setdefault("company", comp)
                    c["metadata"] = meta

                ctx.sort(key=lambda x: (x.get("score") or 0), reverse=True)
                ctx = ctx[: req.top_k]

                # save per-company context for combined summary
                company_ctx_map[comp] = ctx

                gen_result = generate_answer(req.question, ctx, req.conversation_history)
                answer_text = gen_result.get("answer") if isinstance(gen_result, dict) else str(gen_result)
                verification = gen_result.get("verification") if isinstance(gen_result, dict) else []
                sources = build_sources_from_ctx(ctx)

                company_results.append({
                    "company": comp,
                    "answer": answer_text,
                    "sources": [s.dict() for s in sources],
                    "verification": verification,
                })

            # Build a combined, labeled context for a cross-company summary
            combined_ctx: List[Dict[str, Any]] = []
            for comp in req.companies:
                comp_ctx = company_ctx_map.get(comp, [])
                for c in comp_ctx:
                    # Prefix text with company label so the generator can distinguish sources
                    combined_ctx.append(
                        {
                            "text": f"Company: {comp}\n" + (c.get("text") or ""),
                            "metadata": {**(c.get("metadata") or {}), "company": comp},
                            "score": c.get("score"),
                        }
                    )

            # Ask the generator to synthesize a comparative summary across companies
            compare_prompt = (
                f"Compare the following companies: {', '.join(req.companies)} on: {req.question}. "
                "Provide a concise comparative summary that highlights key differences, numeric contrasts where available, and cite which company the evidence comes from. "
                "Be brief and list 3-6 bullet points summarizing the comparison."
            )

            if combined_ctx:
                compare_result = generate_answer(compare_prompt, combined_ctx)
                compare_text = compare_result.get("answer") if isinstance(compare_result, dict) else str(compare_result)
            else:
                compare_text = "No comparative context available to generate a summary."

            # Return the comparative summary as the top-level answer and include per-company details
            return AskResponse(answer=compare_text, sources=[], company_results=company_results)

        # Single company mode (no explicit doc ids)
        if req.company:
            logger.info(f"Processing question for company={req.company}: {req.question[:100]}")
            filters: Dict[str, Any] = {"company": req.company}
            if req.year:
                filters["year"] = req.year

            ctx = retrieve(req.question, top_k=req.top_k, filters=filters)
            if not ctx:
                logger.warning(f"No context found for company: {req.company}")
                return AskResponse(
                    answer="No relevant information found for the specified company.",
                    sources=[],
                    verification=[],
                )

            ctx.sort(key=lambda x: (x.get("score") or 0), reverse=True)
            ctx = ctx[: req.top_k]

            gen_result = generate_answer(req.question, ctx, req.conversation_history)
            answer_text = gen_result.get("answer") if isinstance(gen_result, dict) else str(gen_result)
            verification = gen_result.get("verification") if isinstance(gen_result, dict) else []
            sources = build_sources_from_ctx(ctx)

            logger.info(f"Successfully generated answer for company={req.company}")
            return AskResponse(answer=answer_text, sources=sources, verification=verification)

        # Fallback (shouldn't happen due to validator)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request: could not resolve documents or companies to query",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /ask: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error while processing the question. Please try again.",
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_docs():
    """List all uploaded documents."""
    try:
        docs = list_documents()
        document_infos = [
            DocumentInfo(
                doc_id=doc["doc_id"],
                filename=doc["filename"],
                company=doc.get("company"),
                year=doc.get("year"),
                uploaded_at=doc.get("uploaded_at", ""),
                chunk_count=doc.get("chunk_count", 0),
                file_path=None,  # Don't expose file paths
            )
            for doc in docs
        ]
        
        return DocumentListResponse(
            documents=document_infos,
            total=len(document_infos),
        )
    except Exception as e:
        logger.error(f"Error in /documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document list",
        )


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_doc(doc_id: str):
    """Get information about a specific document."""
    try:
        doc = get_document(doc_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with doc_id '{doc_id}' not found",
            )
        
        return DocumentInfo(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            company=doc.get("company"),
            year=doc.get("year"),
            uploaded_at=doc.get("uploaded_at", ""),
            chunk_count=doc.get("chunk_count", 0),
            file_path=None,  # Don't expose file paths
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /documents/{doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving document",
        )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_doc(doc_id: str):
    """
    Delete a document and its associated data.
    
    Note: This removes metadata, the file, and vectors from FAISS.
    """
    try:
        deleted = delete_document(doc_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with doc_id '{doc_id}' not found",
            )
        
        # Delete vectors from FAISS
        try:
            delete_vectors_by_doc_id(doc_id)
            logger.info(f"Deleted vectors from FAISS for doc_id: {doc_id}")
        except Exception as e:
            logger.warning(f"Error deleting vectors from FAISS: {e}")
        
        logger.info(f"Deleted document: {doc_id}")
        return DeleteResponse(
            doc_id=doc_id,
            message="Document deleted successfully. Vectors removed from FAISS.",
            deleted=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in DELETE /documents/{doc_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document",
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "FinSight RAG Analyst Backend Running",
        "version": "2.0.0",
    }


@app.get("/upload/{upload_id}/progress")
async def get_upload_progress(upload_id: str):
    """
    Get progress for an upload operation.
    
    Returns progress information including:
    - total: Total number of chunks to embed
    - completed: Number of chunks completed
    - percentage: Completion percentage
    - stage: Current stage (parsing, embedding, storing, completed)
    """
    from utils.progress import get_progress
    
    progress = get_progress(upload_id)
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload progress not found. It may have completed or expired.",
        )
    
    return progress

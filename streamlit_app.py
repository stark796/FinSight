# streamlit_app.py

import streamlit as st
import requests
import json
from typing import Optional, Dict, Any
import time
import pandas as pd
import re

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="FinSight - Financial Document Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        line-height: 1.6;
        font-size: 1rem;
    }
    .answer-box p {
        color: #212529;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        color: #212529;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .source-box strong {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(file, company: Optional[str] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """Upload a document to the API."""
    # Reset file pointer to beginning
    file.seek(0)
    
    # Read file content into bytes
    file_bytes = file.read()
    file.seek(0)  # Reset again for good measure
    
    files = {"file": (file.name, file_bytes, "application/pdf")}
    data = {}
    if company:
        data["company"] = company
    if year:
        data["year"] = year
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300  # 5 minutes for large files
        )
        
        # Check if request was successful
        if response.status_code != 201:
            # Try to get error details
            try:
                error_data = response.json()
                error_msg = error_data.get("detail", error_data.get("message", f"HTTP {response.status_code}"))
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
            raise Exception(error_msg)
        
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")

def ask_question(question: str, doc_id: Optional[str] = None, doc_ids: Optional[list] = None, top_k: int = 5, 
                 company: Optional[str] = None, year: Optional[int] = None, companies: Optional[list] = None) -> Dict[str, Any]:
    """Ask a question about one or more documents."""
    payload = {
        "question": question,
        "top_k": top_k
    }
    if doc_ids:
        payload["doc_ids"] = doc_ids
    elif doc_id:
        payload["doc_id"] = doc_id

    if company:
        payload["company"] = company
    if companies:
        payload["companies"] = companies
    if year:
        payload["year"] = year

    response = requests.post(
        f"{API_BASE_URL}/ask",
        json=payload,
        timeout=60
    )

    # Check if request was successful
    if response.status_code != 200:
        try:
            error_data = response.json()
            error_msg = error_data.get("detail", error_data.get("message", f"HTTP {response.status_code}"))
        except:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
        raise Exception(error_msg)


    return response.json()

def get_documents() -> Dict[str, Any]:
    """Get list of all documents."""
    response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to get documents: HTTP {response.status_code}")
    return response.json()

def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document."""
    response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}", timeout=10)
    if response.status_code not in [200, 204]:
        try:
            error_data = response.json()
            error_msg = error_data.get("detail", error_data.get("message", f"HTTP {response.status_code}"))
        except:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
        raise Exception(error_msg)
    return response.json() if response.content else {"deleted": True}

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

# Main app
st.markdown('<div class="main-header">FinSight</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Financial Document Analysis with RAG</div>', unsafe_allow_html=True)

# Check API health
if not check_api_health():
    st.error("⚠️ Cannot connect to the API. Make sure the FastAPI server is running at http://localhost:8000")
    st.info("Start the server with: `uvicorn main:app --reload`")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["Upload Document", "Ask Questions", "Manage Documents"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Refresh documents button
    if st.button("Refresh Documents", use_container_width=True):
        try:
            docs_response = get_documents()
            st.session_state.documents = docs_response.get("documents", [])
            st.success("Documents refreshed!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Error refreshing documents: {e}")

# Load documents on first load
if not st.session_state.documents:
    try:
        docs_response = get_documents()
        st.session_state.documents = docs_response.get("documents", [])
    except Exception as e:
        st.warning(f"Could not load documents: {e}")

# Upload Document Page
if page == "Upload Document":
    st.header("Upload a PDF Document")
    
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a financial document (annual report, earnings statement, etc.)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company Name (optional)", placeholder="e.g., Apple Inc.")
        with col2:
            year = st.number_input("Year (optional)", min_value=2000, max_value=2100, value=None, step=1)
        
        submitted = st.form_submit_button("Upload & Index Document", use_container_width=True)
        
        if submitted and uploaded_file is not None:
            with st.spinner("Uploading and indexing document... This may take a minute."):
                try:
                    result = upload_document(uploaded_file, company if company else None, int(year) if year else None)
                    
                    if result.get("doc_id"):
                        st.success("✅ Document uploaded and indexed successfully!")
                        st.json(result)
                        # Refresh documents list
                        docs_response = get_documents()
                        st.session_state.documents = docs_response.get("documents", [])
                        st.rerun()
                    else:
                        error_msg = result.get('message', 'Unknown error')
                        st.error(f"Upload failed: {error_msg}")
                        st.json(result)  # Show full response for debugging
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. The file might be too large or the server is slow.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API server. Make sure it's running at http://localhost:8000")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {e}")
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Upload failed: {error_msg}")
                    # Show more details in an expander for debugging
                    with st.expander("Error Details"):
                        st.code(str(e), language="text")
        elif submitted:
            st.warning("Please select a PDF file to upload.")

# Ask Questions Page
elif page == "Ask Questions":
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.documents:
        st.info("No documents uploaded yet. Go to 'Upload Document' to add a document first.")
    else:
        # Document selector (supports multi-select for cross-doc queries)
        doc_options = {doc['doc_id']: f"{doc['filename']} ({doc.get('company', 'N/A')} - {doc.get('year', 'N/A')})" for doc in st.session_state.documents}
        selected_doc_ids = st.multiselect("Select one or more documents (ignored in company modes)", options=list(doc_options.keys()), format_func=lambda k: doc_options[k])

        # Company options derived from uploaded documents
        company_names = sorted(list({doc.get('company') for doc in st.session_state.documents if doc.get('company')}))

        # Query mode: allow Document(s), Company, or Compare Companies
        query_mode = st.radio("Query Mode", ["By Document(s)", "By Company", "Compare Companies"], index=0)

        selected_company = None
        compare_companies = None

        if query_mode == "By Document(s)":
            if selected_doc_ids:
                if len(selected_doc_ids) == 1:
                    selected_doc = next(doc for doc in st.session_state.documents if doc['doc_id'] == selected_doc_ids[0])
                    with st.expander("📄 Document Information", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Filename", selected_doc['filename'])
                        with col2:
                            st.metric("Chunks", selected_doc.get('chunk_count', 0))
                        with col3:
                            st.metric("Uploaded", selected_doc.get('uploaded_at', 'N/A')[:10] if selected_doc.get('uploaded_at') else 'N/A')
                else:
                    st.info(f"Selected {len(selected_doc_ids)} documents for cross-document query.")

        elif query_mode == "By Company":
            selected_company = st.selectbox("Select a company", options=[""] + company_names)
            if selected_company == "":
                selected_company = None

        else:  # Compare Companies
            compare_companies = st.multiselect("Select companies to compare (choose 2+)", options=company_names)

        st.divider()

        # Question input (always visible)
        question = st.text_area(
            "Enter your question",
            height=100,
            placeholder="e.g., What was the revenue growth? What are the key risks mentioned?",
            help="Ask any question about the content of the document"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=20, value=10, help="More sources = more context. Recommended: 10-15 for better answers")
        with col2:
            ask_button = st.button("🔍 Ask Question", use_container_width=True, type="primary")

        if ask_button and question.strip():
            with st.spinner("Searching document and generating answer..."):
                try:
                    # Decide payload based on query mode
                    if query_mode == "By Document(s)":
                        if selected_doc_ids and len(selected_doc_ids) > 1:
                            result = ask_question(
                                question,
                                doc_ids=selected_doc_ids,
                                top_k=top_k,
                            )
                        elif selected_doc_ids and len(selected_doc_ids) == 1:
                            sel = next(doc for doc in st.session_state.documents if doc['doc_id'] == selected_doc_ids[0])
                            result = ask_question(
                                question,
                                doc_id=selected_doc_ids[0],
                                top_k=top_k,
                                company=sel.get('company'),
                                year=sel.get('year'),
                            )
                        else:
                            st.warning("Please select one or more documents to query.")
                            st.stop()

                    elif query_mode == "By Company":
                        if not selected_company:
                            st.warning("Please select a company to query.")
                            st.stop()
                        result = ask_question(
                            question,
                            top_k=top_k,
                            company=selected_company,
                        )

                    else:  # Compare Companies
                        if not compare_companies or len(compare_companies) < 2:
                            st.warning("Select two or more companies to compare.")
                            st.stop()
                        result = ask_question(
                            question,
                            top_k=top_k,
                            companies=compare_companies,
                        )

                    # Display answer(s)
                    st.markdown("### Answer")
                    answer_text = result.get("answer", "No answer generated.")
                    st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)

                    # If API returned per-company results (comparison), render those
                    company_results = result.get("company_results") or []
                    if company_results:
                        st.markdown("### Per-Company Results")
                        for comp_res in company_results:
                            comp = comp_res.get("company")
                            st.subheader(f"{comp}")
                            st.markdown(f'<div class="answer-box">{comp_res.get("answer","No answer")}</div>', unsafe_allow_html=True)
                            sources = comp_res.get("sources", [])
                            if sources:
                                st.markdown("#### Sources")
                                for i, source in enumerate(sources, 1):
                                    with st.container():
                                        col1, col2 = st.columns([1, 4])
                                        with col1:
                                            if source.get("page"):
                                                st.metric("Page", source["page"])
                                            if source.get("score"):
                                                st.caption(f"Score: {source['score']:.3f}")
                                        with col2:
                                            rows = source.get("rows") or (source.get("metadata") or {}).get("rows")
                                            chunk_type = source.get("chunk_type") or (source.get("metadata") or {}).get("chunk_type")
                                            if rows and isinstance(rows, list) and len(rows) > 0:
                                                try:
                                                    if isinstance(rows[0], (list, tuple)):
                                                        header = [str(h) for h in rows[0]]
                                                        data = rows[1:] if len(rows) > 1 else []
                                                        df = pd.DataFrame(data, columns=header)
                                                        st.markdown(f"**Source {i} — Table (page {source.get('page', 'N/A')})**")
                                                        st.table(df)
                                                    else:
                                                        st.markdown(f'<div class="source-box"><strong>Source {i}</strong><br>{source.get("snippet", source.get("text", "No snippet available"))}</div>', unsafe_allow_html=True)
                                                except Exception:
                                                    st.markdown(f'<div class="source-box"><strong>Source {i}</strong><br>{source.get("snippet", source.get("text", "No snippet available"))}</div>', unsafe_allow_html=True)
                    else:
                        # Legacy single-result rendering
                        sources = result.get("sources", [])
                        if sources:
                            st.markdown("### Sources")
                            for i, source in enumerate(sources, 1):
                                with st.container():
                                    col1, col2 = st.columns([1, 4])
                                    with col1:
                                        if source.get("page"):
                                            st.metric("Page", source["page"])
                                        if source.get("score"):
                                            st.caption(f"Score: {source['score']:.3f}")
                                    with col2:
                                        rows = source.get("rows") or (source.get("metadata") or {}).get("rows")
                                        chunk_type = source.get("chunk_type") or (source.get("metadata") or {}).get("chunk_type")
                                        if rows and isinstance(rows, list) and len(rows) > 0:
                                            try:
                                                if isinstance(rows[0], (list, tuple)):
                                                    header = [str(h) for h in rows[0]]
                                                    data = rows[1:] if len(rows) > 1 else []
                                                    df = pd.DataFrame(data, columns=header)
                                                    st.markdown(f"**Source {i} — Table (page {source.get('page', 'N/A')})**")
                                                    st.table(df)
                                                else:
                                                    st.markdown(f'<div class="source-box"><strong>Source {i}</strong><br>{source.get("snippet", source.get("text", "No snippet available"))}</div>', unsafe_allow_html=True)
                                            except Exception:
                                                st.markdown(f'<div class="source-box"><strong>Source {i}</strong><br>{source.get("snippet", source.get("text", "No snippet available"))}</div>', unsafe_allow_html=True)
                        else:
                            st.info("No sources found for this question.")

                    # Display verification results if present
                    verification = result.get("verification") or []
                    if verification:
                        st.markdown("### Numeric Verification")
                        for v in verification:
                            matched = v.get("matched")
                            claim = v.get("claim_token")
                            src = v.get("source_token")
                            rel = v.get("rel_error")
                            if matched:
                                st.success(f"Claim {claim} matches source {src} (rel error {rel:.3f})")
                            else:
                                st.warning(f"Claim {claim} NOT matched in sources")

                    # Display LLM fact-check results if present
                    fact_checks = result.get("fact_check") or []
                    if fact_checks:
                        st.markdown("### LLM Fact-Check")
                        for fc in fact_checks:
                            claim = fc.get("claim") or fc.get("claim_text") or "(claim)"
                            verdict = fc.get("verdict")
                            sources = fc.get("sources") or []
                            explanation = fc.get("explanation") or ""
                            if verdict == "supported":
                                st.success(f"{claim} — SUPPORTED by sources {sources}")
                            elif verdict == "contradicted":
                                st.error(f"{claim} — CONTRADICTED by sources {sources}")
                            else:
                                st.warning(f"{claim} — NOT SUPPORTED by provided context")
                            if explanation:
                                with st.expander("Explanation"):
                                    st.write(explanation)
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The question might be too complex or the server is slow.")
                except Exception as e:
                    st.error(f"Error asking question: {e}")

# Manage Documents Page
elif page == "Manage Documents":
    st.header("Manage Documents")
    
    if not st.session_state.documents:
        st.info("No documents uploaded yet.")
    else:
        st.metric("Total Documents", len(st.session_state.documents))
        st.divider()
        
        # Display documents in a table
        for doc in st.session_state.documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.write(f"**{doc['filename']}**")
                    if doc.get('company'):
                        st.caption(f"Company: {doc.get('company')}")
                
                with col2:
                    st.caption(f"Uploaded: {doc.get('uploaded_at', 'N/A')[:19] if doc.get('uploaded_at') else 'N/A'}")
                    st.caption(f"Chunks: {doc.get('chunk_count', 0)}")
                
                with col3:
                    if doc.get('year'):
                        st.caption(f"Year: {doc.get('year')}")
                
                with col4:
                    if st.button("Delete", key=f"delete_{doc['doc_id']}", type="secondary"):
                        with st.spinner("Deleting..."):
                            try:
                                result = delete_document(doc['doc_id'])
                                if result.get("deleted"):
                                    st.success(f"Deleted: {doc['filename']}")
                                    # Refresh documents
                                    docs_response = get_documents()
                                    st.session_state.documents = docs_response.get("documents", [])
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {result.get('message', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"Error deleting document: {e}")
                
                st.divider()

# Footer
st.divider()
st.caption("FinSight v2.0.0 | Powered by Google Gemini & Pinecone")


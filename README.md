# FinSight

A RAG system for analyzing financial documents. Upload PDFs, ask questions, get answers with citations. Uses Google Gemini for embeddings and generation, Pinecone for vector storage.

## What It Does

You can upload financial PDFs (like annual reports, earnings statements, etc.) and then ask questions about them. The system will:
- Parse and chunk the PDFs intelligently
- Store embeddings in Pinecone for fast semantic search
- Answer your questions using the document content
- Show you where the answers came from (page numbers, snippets)

## What You Need

- Python 3.9 or higher
- A Google Gemini API key (get one from Google AI Studio)
- A Pinecone account and API key
- A Pinecone index (create one in your Pinecone dashboard)

## Setup

First, install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name
ENV=dev
```

The app will create these directories automatically when you run it:
- `uploads/` - where uploaded PDFs are stored
- `logs/` - application logs
- `data/` - document metadata

## Running It

### Option 1: FastAPI Backend Only

Start the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then visit:
- http://localhost:8000/docs for interactive API documentation
- http://localhost:8000/redoc for alternative docs
- http://localhost:8000/health to check if it's running

### Option 2: Streamlit Web App (Recommended)

The Streamlit app provides a user-friendly interface for uploading documents and asking questions.

1. First, make sure the FastAPI server is running (see Option 1 above)

2. In a new terminal, start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Or use the provided script:

```bash
./run_streamlit.sh
```

3. The Streamlit app will open in your browser at http://localhost:8501

The Streamlit app has three main pages:
- **Upload Document**: Upload PDFs with optional metadata
- **Ask Questions**: Select a document and ask questions about it
- **Manage Documents**: View and delete uploaded documents

## API Endpoints

### Upload a Document

POST `/upload`

Upload a PDF file. You can optionally include company name and year as metadata.

Form data:
- `file`: The PDF file (required, max 50MB)
- `company`: Company name (optional)
- `year`: Year (optional)

Returns a `doc_id` that you'll use to ask questions about this document.

Example response:
```json
{
  "doc_id": "abc-123-def",
  "filename": "annual_report_2024.pdf",
  "company": "Example Corp",
  "year": 2024,
  "message": "File uploaded and indexed successfully.",
  "file_size": 1234567
}
```

### Ask a Question

POST `/ask`

Ask a question about a document you've uploaded.

Request body:
```json
{
  "question": "What was the revenue growth?",
  "top_k": 5,
  "doc_id": "abc-123-def",
  "company": "Example Corp",
  "year": 2024
}
```

The `top_k` parameter controls how many document chunks to retrieve (default is 5, max is 20).

Response includes the answer and source citations:
```json
{

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
  "answer": "The revenue grew by 25% year-over-year...",
  "sources": [
    {
      "page": 5,
      "score": 0.92,
      "snippet": "Revenue increased from $100M to $125M..."
    }
  ]
}
```

### List All Documents

GET `/documents`

Get a list of all documents you've uploaded, with metadata like upload time and chunk count.

### Get Document Info

GET `/documents/{doc_id}`

Get details about a specific document.

### Delete a Document

DELETE `/documents/{doc_id}`

Delete a document and its metadata. Note: This doesn't remove the vectors from Pinecone - you'd need to implement a cleanup job for that if you want full deletion.

### Health Check

GET `/health`

Simple endpoint to check if the API is running.

## How It Works

The code is organized like this:

```
main.py                    # FastAPI app and endpoints
├── rag_pipeline/
│   ├── ingest.py         # PDF parsing and chunking
│   ├── embed_store.py    # Creating embeddings and storing in Pinecone
│   ├── retrieve.py       # Searching for relevant chunks
│   └── generate.py       # Generating answers with Gemini
└── utils/
    ├── config.py         # Configuration from environment
    ├── logger.py         # Logging setup
    ├── retry.py          # Retry logic for API calls
    └── document_store.py # Tracking document metadata
```

## Features

**Better chunking**: Instead of just splitting text at fixed character counts, it tries to split at sentence boundaries and includes some overlap between chunks so context isn't lost.

**Error handling**: Proper HTTP status codes, retry logic for API calls (with exponential backoff), and cleanup if something goes wrong during upload.

**Logging**: All operations are logged to both console and `logs/app.log` so you can debug issues.

**Validation**: File size limits (50MB), type checking, and input validation to catch problems early.

**Document management**: You can list, view details, and delete documents through the API.

## Configuration

You can tweak settings in `utils/config.py`:
- `CHUNK_SIZE`: How many characters per chunk (default 1200)
- `TOP_K_DEFAULT`: Default number of chunks to retrieve (default 5)
- `EMBED_MODEL`: Which Gemini model to use for embeddings
- `GENERATION_MODEL`: Which Gemini model to use for generating answers

## Security Notes

**CORS**: Right now it allows all origins. For production, you should update the CORS settings in `main.py` to only allow your frontend domain.

**File storage**: Files are stored locally in the `uploads/` directory. For production, consider using cloud storage like S3 or Google Cloud Storage.

**API keys**: Never commit your `.env` file. Use environment variables or a secrets manager in production.

## Troubleshooting

**Import errors**: Make sure you're running from the project root directory and all dependencies are installed.

**Pinecone issues**: 
- Double-check your API key and index name in `.env`
- Make sure the index exists in your Pinecone dashboard
- The index dimension should match your embedding model (768 for text-embedding-004)

**Gemini API errors**:
- Verify your API key is correct
- Check if you've hit rate limits
- Look at `logs/app.log` for detailed error messages

## Things That Could Be Added

- Async embedding operations to speed things up
- Actually deleting vectors from Pinecone when a document is deleted
- Rate limiting
- Authentication/authorization
- Batch uploads
- Support for other file types (Word docs, plain text)
- Proper database instead of JSON file for metadata
- Cloud storage integration
- Tests

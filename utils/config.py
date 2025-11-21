# utils/config.py

import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ----------------------------
# GEMINI (Google Generative AI)
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")

# Embeddings model
# Highly recommended for documents
EMBED_MODEL = "models/text-embedding-004"

# Generation model
# Available models: gemini-2.5-flash (fast), gemini-2.5-pro (more capable)
# You can also use "models/gemini-flash-latest" for always latest flash model
GENERATION_MODEL = "models/gemini-2.5-flash"

# ----------------------------
# PINECONE
# ----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
    raise ValueError("Missing PINECONE_API_KEY in environment variables")

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "<your-index-name>")

# ----------------------------
# Debug / Optional Settings
# ----------------------------
CHUNK_SIZE = 1200               # number of characters per chunk
TOP_K_DEFAULT = 5               # default number of retrieved chunks
ENVIRONMENT = os.getenv("ENV", "dev")


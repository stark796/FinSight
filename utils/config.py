import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_api_key")

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

PINECONE_INDEX = "finsight-rag"

import uuid
from openai import OpenAI
from pinecone import Pinecone
from utils.config import OPENAI_API_KEY, EMBED_MODEL, PINECONE_API_KEY, PINECONE_INDEX

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_chunks(chunks: list):
    """Create embeddings for chunks."""
    vectors = []
    for chunk in chunks:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk
        )
        vectors.append(response.data[0].embedding)
    return vectors

def store_chunks(chunks: list, embeddings: list):
    """Store chunks & embeddings in Pinecone vector DB."""
    items = []
    for chunk, vector in zip(chunks, embeddings):
        items.append((str(uuid.uuid4()), vector, {"text": chunk}))

    index.upsert(items)
    print(f"[✓] Stored {len(items)} chunks in vector index.")

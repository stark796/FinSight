from openai import OpenAI
from pinecone import Pinecone
from utils.config import OPENAI_API_KEY, EMBED_MODEL, PINECONE_API_KEY, PINECONE_INDEX

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def retrieve(query: str, top_k: int = 5):
    """Retrieve top-k chunk texts."""
    # Embed query
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    # Query vector DB
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )

    return [m["metadata"]["text"] for m in results["matches"]]

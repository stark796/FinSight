from openai import OpenAI
from utils.config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(query: str, context_chunks: list):
    """Generate answer using RAG."""
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are Finsight AI, a financial analysis assistant.
Answer only using the context below.

Context:
{context}

Query:
{query}

Provide a clear and concise answer.
"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

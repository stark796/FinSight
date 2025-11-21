# generate.py

import google.generativeai as genai
from utils.config import GEMINI_API_KEY, GENERATION_MODEL
from utils.logger import logger
from utils.retry import retry_with_backoff
from utils.numeric import compare_claims_to_context
import json

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """
You are a senior financial analyst helping users understand financial documents.

Guidelines:
1. Use the information provided in the context to answer the question.
2. The context may include both text and tables. Tables are formatted as markdown-style tables with rows and columns.
3. When reading tables, pay attention to column headers and row labels to understand the data structure.
4. You can infer and synthesize information from the context, but base your answer on what's actually there.
5. If the context contains relevant information (even if not a direct quote), provide a helpful answer.
6. Only say "The document does not contain enough information" if the context truly has nothing relevant.
7. For numbers, percentages, and calculations:
     - Extract numbers FROM THE CONTEXT (including from tables)
     - Show step-by-step calculations when possible
     - Clearly state if you're inferring or estimating
8. Always cite sources using [Source X] format.
9. Be helpful and informative while staying factual.
"""


def build_context(context_chunks: list[dict]) -> str:
    """
    Turns retrieved chunks into a numbered source list.
    """
    parts = []
    for i, c in enumerate(context_chunks):
        header = f"[Source {i+1}]"
        meta = c.get("metadata", {}) or {}
        # If chunk is a table, label it and include the markdown table text
        if meta.get("chunk_type") == "table":
            tbl_label = f"[Source {i+1} — Table (page {meta.get('page', 'N/A')})]"
            parts.append(f"{tbl_label}\n{c.get('text','')}")
        else:
            parts.append(f"{header}\n{c.get('text','')}")
    return "\n\n".join(parts)


def fact_check_answer(answer: str, context_chunks: list[dict]) -> list:
    """Use the LLM to fact-check the answer against the provided context chunks.

    Returns a list of dicts: {claim, verdict, sources, explanation}
    where verdict is one of: supported, contradicted, unsupported.
    """
    logger.info("Running LLM fact-check for generated answer")

    # Build a compact context string
    context_text = build_context(context_chunks)

    prompt = (
        "You are a fact-checking assistant.\n"
        "Given the generated ANSWER and the CONTEXT below, identify distinct factual claims made in the ANSWER.\n"
        "For each claim, determine whether it is 'supported', 'contradicted', or 'unsupported' by the CONTEXT.\n"
        "If supported, list the source indices (e.g., Source 1) that support it. If contradicted, list the sources that contradict.\n"
        "Return the result as JSON: an array of objects {\n"
        "  \"claim\": string,\n"
        "  \"verdict\": \"supported\"|\"contradicted\"|\"unsupported\",\n"
        "  \"sources\": [integers],\n"
        "  \"explanation\": string\n"
        "}\n"
        "Only output valid JSON. Use the context only; do not consult external knowledge.\n\n"
        f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}\n\nJSON:\n"
    )

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        text = response.text
        # Try to parse JSON from the model output
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            else:
                return [parsed]
        except Exception:
            # If parsing fails, return raw text as single-item explanation
            return [{"claim": "<raw>", "verdict": "unsupported", "sources": [], "explanation": text}]
    except Exception as e:
        logger.error(f"Error during fact_check_answer: {e}", exc_info=True)
        return []


@retry_with_backoff(max_retries=3, exceptions=(Exception,))
def generate_answer(query: str, context_chunks: list[dict]) -> dict:
    """
    Generate an answer using Gemini, constrained to provided context.
    Returns a dict with keys: 'answer' (str) and 'verification' (list) where verification
    contains numeric claim checks against the provided context.
    """
    logger.info(f"Generating answer for query: {query[:100]}...")
    
    if not context_chunks:
        logger.warning("No context chunks provided for answer generation")
        return "No relevant context found to generate an answer."
    
    context_text = build_context(context_chunks)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context from document:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Instructions:\n"
        "- Answer the question based on the context provided above.\n"
        "- If the context contains relevant information, provide a helpful answer even if it's not a direct quote.\n"
        "- Cite your sources using [Source X] format.\n"
        "- If the context truly has no relevant information, then say the document doesn't contain enough information.\n"
        "- Be specific and cite page numbers when available.\n"
    )

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(full_prompt)

        # Gemini returns a response object; .text gives final combined output
        answer = response.text
        logger.info("Successfully generated answer")

        # Run numeric verification: compare numeric mentions in the answer to context chunks
        context_texts = [c.get("text", "") for c in context_chunks]
        verification = compare_claims_to_context(answer, context_texts)

        # Run LLM-based fact-check: ask the model to classify claims as supported/contradicted/unsupported
        fact_check = []
        try:
            fact_check = fact_check_answer(answer, context_chunks)
        except Exception as e:
            logger.warning(f"Fact-check failed: {e}")

        return {"answer": answer, "verification": verification, "fact_check": fact_check}

    except Exception as e:
        logger.error(f"Error in generate_answer(): {e}", exc_info=True)
        return {"answer": "Error generating answer using Gemini. Please try again.", "verification": []}

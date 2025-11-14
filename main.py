from rag_pipeline.ingest import extract_text, chunk_text
from rag_pipeline.embed_store import embed_chunks, store_chunks
from rag_pipeline.retrieve import retrieve
from rag_pipeline.generate import generate_answer

def main():
    print("=== Finsight RAG ===")
    pdf = input("Enter PDF path: ")

    print("\n[1] Extracting text...")
    text = extract_text(pdf)

    print("[2] Chunking...")
    chunks = chunk_text(text)

    print("[3] Embedding + Storing...")
    vectors = embed_chunks(chunks)
    store_chunks(chunks, vectors)

    print("[✓] Ingestion complete.")
    print("\nAsk questions about the report (type 'exit' to quit)\n")

    while True:
        q = input("> ")
        if q.lower() == "exit":
            break

        context = retrieve(q)
        answer = generate_answer(q, context)

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")

if __name__ == "__main__":
    main()

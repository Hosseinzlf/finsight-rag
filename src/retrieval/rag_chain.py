import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, Optional

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings

# Load the same model and collection we used in Phase 2
model  = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="data/embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name="filings")

openai = OpenAI()

def retrieve(query: str, n_results: int = 5, ticker: Optional[str] = None) -> dict[str, Any]:
    query_embedding = model.encode(query).tolist()
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if ticker:
        query_kwargs["where"] = {"ticker": ticker}

    return collection.query(**query_kwargs)

def ask(question: str, ticker: Optional[str] = None) -> dict[str, Any]:
    # Step 1 — find relevant chunks
    results = retrieve(question, ticker=ticker)
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not chunks:
        return {
            "answer": "I don't have enough information.",
            "sources": [],
            "tickers_searched": [ticker] if ticker else ["ALL"],
        }

    # Step 2 — build the prompt
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a financial analyst assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}
Answer:"""

    # Step 3 — send to LLM
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    answer = response.choices[0].message.content
    sources = []
    tickers_found = set()
    for i, chunk_text in enumerate(chunks):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        source = {
            "source": metadata.get("source", "Unknown"),
            "ticker": metadata.get("ticker", "UNKNOWN"),
            "company": metadata.get("company", "Unknown"),
            "chunk_index": metadata.get("chunk_index"),
            "text": chunk_text,
        }
        tickers_found.add(source["ticker"])
        sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "tickers_searched": [ticker] if ticker else sorted(tickers_found) or ["ALL"],
    }

if __name__ == "__main__":
    questions = [
        "What are Apple's main risk factors?",
        "How much revenue did Tesla make in 2024?",
        "What did Amazon say about AI in their filing?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 50)







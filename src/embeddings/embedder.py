# src/embeddings/embedder.py

import json
import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load the embedding model — runs locally, completely free
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add this line when creating the client
client = chromadb.PersistentClient(
    path="data/embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_or_create_collection(name="filings")

def embed_chunks(chunks_path: Path):
    with open(chunks_path) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    ids   = [c["chunk_id"] for c in chunks]
    metas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"Embedding {len(texts)} chunks... (this takes 1-2 minutes)")

    # Embed in batches of 64 — avoids memory issues
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids   = ids[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]

        embeddings = model.encode(batch_texts).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metas,
        )

        print(f"  {min(i+batch_size, len(texts))}/{len(texts)} done")

    print(f"All chunks embedded and saved to ChromaDB.")

def test_search(query: str):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
    )
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc[:100])
        print("...")

if __name__ == "__main__":
    chunks_dir = Path("data/chunks")

    for path in chunks_dir.glob("*_chunks.json"):
        print(f"Processing {path.name}...")
        embed_chunks(path)

    # Test it works
    test_search("How much revenue did Tesla make?")
    test_search("How much revenue did Apple make?")
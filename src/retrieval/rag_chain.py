import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load the same model and collection we used in Phase 2
model  = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="data/embeddings")
collection = client.get_or_create_collection(name="filings")

openai = OpenAI()

def retrieve(query: str, n_results: int = 5) -> list[str]:
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0]

def ask(question: str) -> str:
    # Step 1 — find relevant chunks
    chunks = retrieve(question)

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

    return response.choices[0].message.content

if __name__ == "__main__":
    questions = [
        "What are Apple's main risk factors?",
        "How much revenue did Apple make in 2024?",
        "What did Apple say about AI in their filing?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 50)
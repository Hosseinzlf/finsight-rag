# src/evaluation/metrics.py

import sys
sys.path.append(".")

from src.retrieval.rag_chain import retrieve, ask

TEST_CASES = [
    {
        "question": "What was Apple's total net sales in 2024?",
        "keywords": ["391", "million", "sales", "revenue", "total"]
    },
    {
        "question": "What are Apple's main risk factors?",
        "keywords": ["competition", "supply", "regulation", "risk", "market"]
    },
    {
        "question": "What products did Apple announce in 2024?",
        "keywords": ["macbook", "iphone", "ipad", "announced", "quarter"]
    },
]

def evaluate_faithfulness(answer: str, chunks: list[str]) -> float:
    if "don't have enough" in answer.lower():
        return 0.0

    chunk_text = " ".join(chunks).lower()

    # Short answer handling
    if len(answer) < 100:
        # Strip punctuation for matching
        import re
        clean_answer = re.sub(r'[,$.]', '', answer.lower())
        clean_chunks = re.sub(r'[,$.]', '', chunk_text)
        words = [w for w in clean_answer.split() if len(w) > 2]
        matches = sum(1 for w in words if w in clean_chunks)
        return round(matches / len(words), 2) if words else 0.0

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        words = [w for w in sentence.lower().split() if len(w) > 3]
        matches = sum(1 for w in words if w in chunk_text)
        if matches >= 2:
            grounded += 1

    return round(grounded / len(sentences), 2)

def evaluate_relevance(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw in answer_lower)
    return round(matches / len(keywords), 2)

def run_evaluation():
    print("Running evaluation...\n")
    print(f"{'Question':<45} {'Faithfulness':>12} {'Relevance':>10}")
    print("-" * 70)

    total_faith = 0
    total_rel   = 0

    for case in TEST_CASES:
        question = case["question"]
        keywords = case["keywords"]

        answer = ask(question)
        chunks = retrieve(question)

        faith = evaluate_faithfulness(answer, chunks)
        rel   = evaluate_relevance(answer, keywords)

        total_faith += faith
        total_rel   += rel

        print(f"{question:<45} {faith:>12.2f} {rel:>10.2f}")
        print(f"  Answer: {answer[:120]}...")
        print()

    n = len(TEST_CASES)
    print("-" * 70)
    print(f"{'Average':<45} {total_faith/n:>12.2f} {total_rel/n:>10.2f}")
    print("\nScores are 0 to 1. Above 0.7 is good for a first version.")

if __name__ == "__main__":
    run_evaluation()
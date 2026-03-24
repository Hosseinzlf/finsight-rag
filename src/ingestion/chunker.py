import json
import re
from html import unescape
from pathlib import Path


TICKER_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
}

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def clean_text(text: str) -> str:
    # Remove hidden Inline XBRL metadata block (high-noise, low-value for QA)
    text = re.sub(r"<ix:hidden>.*?</ix:hidden>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove HTML/XBRL tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities like &#8217;
    text = unescape(text)
    # Remove taxonomy/namespace tokens and URLs (high-noise for semantic search)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b[\w.-]+:[\w.#-]+\b", " ", text)
    # Basic whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()
    return text

def process_file(filepath, output_dir):
    text = filepath.read_text(encoding="utf-8")
    text = clean_text(text)
    
    chunks = chunk_text(text)
    
    # Extract ticker from filename e.g. "AAPL_2024_10K.txt" → "AAPL"
    ticker = filepath.stem.split("_")[0]
    company = TICKER_MAP.get(ticker, ticker)  # fallback to ticker if not in map
    
    result = []
    for i, chunk in enumerate(chunks):
        if len(re.findall(r"[A-Za-z]{3,}", chunk)) < 30:
            continue
        result.append({
            "chunk_id": f"{filepath.stem}_{i:04d}",
            "text": chunk,
            "source": filepath.name,
            "chunk_index": i,
            "ticker": ticker,       # ← NEW
            "company": company,     # ← NEW
        })
    
    out_path = output_dir / f"{filepath.stem}_chunks.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Done: {len(result)} chunks saved to {out_path.name} [{ticker}]")
    return result

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(exist_ok=True)

    for filepath in raw_dir.glob("*.txt"):
        print(f"Processing {filepath.name}...")
        process_file(filepath, chunks_dir)
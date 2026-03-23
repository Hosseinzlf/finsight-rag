import json
from pathlib import Path

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_file(filepath, output_dir):
    text = filepath.read_text(encoding="utf-8")
    
    # Basic cleaning
    import re
    text = re.sub(r'\n{4,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    
    chunks = chunk_text(text)
    
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "chunk_id": f"{filepath.stem}_{i:04d}",
            "text": chunk,
            "source": filepath.name,
            "chunk_index": i
        })
    
    out_path = output_dir / f"{filepath.stem}_chunks.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Done: {len(result)} chunks saved to {out_path.name}")
    return result

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(exist_ok=True)

    for filepath in raw_dir.glob("*.txt"):
        print(f"Processing {filepath.name}...")
        process_file(filepath, chunks_dir)
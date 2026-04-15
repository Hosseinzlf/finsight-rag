# FinSight: Financial Filing Q&A with RAG

FinSight is an end-to-end Retrieval-Augmented Generation (RAG) project for asking questions about SEC-style financial filings.  
It ingests raw filing text, chunks and cleans it, builds local embeddings with ChromaDB, and serves grounded answers through a FastAPI backend and Streamlit UI.

The current implementation is intentionally lightweight and practical: local retrieval, simple prompt grounding, and a minimal evaluation script for quick iteration.
 
## What This Project Does

- Ingests filing text from `data/raw/`
- Cleans high-noise markup/taxonomy artifacts
- Splits filings into overlapping chunks
- Embeds chunks using `all-MiniLM-L6-v2`
- Stores vectors in persistent ChromaDB (`data/embeddings/`)
- Retrieves top relevant chunks for a user question
- Generates an answer with OpenAI chat completion
- Exposes:
  - FastAPI endpoint at `/ask`
  - Streamlit app for interactive Q&A

## Repository Structure

```text
finsight/
├── app.py                          # Streamlit frontend
├── requirements.txt
├── data/
│   ├── raw/                        # Raw filing text input
│   ├── chunks/                     # Chunked JSON output
│   └── embeddings/                 # Persistent ChromaDB files
└── src/
    ├── api/main.py                 # FastAPI app
    ├── ingestion/chunker.py        # Text cleaning + chunking
    ├── embeddings/embedder.py      # Local embedding + indexing
    ├── retrieval/rag_chain.py      # Retrieval + answer generation
    └── evaluation/metrics.py       # Basic faithfulness/relevance metrics
```

## Architecture (High Level)

1. **Ingestion**: read raw filing text and normalize/clean it  
2. **Chunking**: split text into overlapping windows (`1000` chars, `200` overlap)  
3. **Embedding**: encode chunks with SentenceTransformers (`all-MiniLM-L6-v2`)  
4. **Vector Storage**: persist vectors in ChromaDB collection `filings`  
5. **Retrieval**: semantic similarity search returns top matching chunks  
6. **Generation**: question + retrieved context sent to OpenAI for grounded answer  
7. **Serving**: FastAPI endpoint consumed by Streamlit UI

## Requirements

- Python 3.10+ (3.11 recommended)
- An OpenAI API key for answer generation

## Quick Start

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file in `finsight/`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4) Add filing text

Place one or more `.txt` files into:

```text
data/raw/
```

Example already present in this repo:

- `data/raw/AAPL_2024_10K.txt`

### 5) Run ingestion (clean + chunk)

```bash
python src/ingestion/chunker.py
```

Generated files are written to:

```text
data/chunks/*_chunks.json
```

### 6) Build embeddings and index

```bash
python src/embeddings/embedder.py
```

This writes vectors to:

```text
data/embeddings/
```
 
### 7) Start API backend

```bash
uvicorn src.api.main:app --reload
```

API default:

- Base URL: `http://localhost:8000`
- Health route: `GET /`
- Q&A route: `POST /ask`

### 8) Start Streamlit frontend (separate terminal)
 
```bash
streamlit run app.py
```

Open the provided local Streamlit URL and ask questions about the indexed filings.


## Evaluation

Run the lightweight evaluator:

```bash
python src/evaluation/metrics.py
```

Current script reports two simple heuristics:

- **Faithfulness**: how much answer content is grounded in retrieved text
- **Relevance**: keyword overlap with expected topic terms

This is useful for fast iteration, but not a replacement for a formal benchmark.


import sys
sys.path.append(".")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.retrieval.rag_chain import ask

app = FastAPI(title="FinSight API")

class Question(BaseModel):
    text: str
    ticker: Optional[str] = None  # ← NEW: optional company filter from UI

@app.get("/")
def root():
    return {"status": "ok", "message": "FinSight API is running"}

@app.post("/ask")
def ask_question(question: Question):
    # ask() now returns a dict with answer + sources + tickers_searched
    result = ask(question.text, question.ticker)
    return {
        "question": question.text,
        "answer": result["answer"],
        "sources": result["sources"],
        "tickers_searched": result["tickers_searched"],
    }
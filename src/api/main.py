import sys
sys.path.append(".")

from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.rag_chain import ask, retrieve

app = FastAPI(title="FinSight API")

class Question(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "FinSight API is running"}

@app.post("/ask")
def ask_question(question: Question):
    answer = ask(question.text)
    chunks = retrieve(question.text)
    return {
        "question": question.text,
        "answer": answer,
        "sources": chunks[:1]
    }
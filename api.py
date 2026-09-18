from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import shutil
import rag_core

app = FastAPI(title="Veris API")

class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(payload: Question):
    return rag_core.ask_question(payload.question)

@app.post("/absorb")
def absorb(file: UploadFile):
    filepath = f"./{file.filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    num_chunks = rag_core.ingest_pdf(filepath, doc_id_prefix=file.filename)
    return {"filename": file.filename, "chunks_added": num_chunks}
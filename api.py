from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
import shutil
import rag_core

app = FastAPI(title="Veris API")

class Question(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
def ask(payload: Question):
    return rag_core.ask_question(payload.question, payload.session_id)

@app.post("/absorb")
def absorb(file: UploadFile, session_id: str = Form(...)):
    filepath = f"./{session_id}_{file.filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    num_chunks = rag_core.ingest_pdf(filepath, session_id, doc_id_prefix=file.filename)
    return {"filename": file.filename, "chunks_added": num_chunks}
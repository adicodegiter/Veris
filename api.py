from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel
import os
from uuid import UUID
import rag_core

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Veris API")

class Question(BaseModel):
    question: str
    session_id: str

def _profile_id(value):
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid browser profile")

@app.post("/ask")
def ask(payload: Question):
    return rag_core.ask_question(payload.question, _profile_id(payload.session_id))

@app.get("/documents")
def documents(session_id: str):
    return {"documents": rag_core.list_documents(_profile_id(session_id))}

@app.post("/absorb")
def absorb(file: UploadFile, session_id: str = Form(...)):
    profile_id = _profile_id(session_id)
    filename = os.path.basename((file.filename or "").replace("\\", "/"))
    extension = os.path.splitext(filename)[1].lower()
    if extension not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Unsupported document type")
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Documents must be 10 MB or smaller")
    num_chunks = rag_core.ingest_document_bytes(file_bytes, profile_id, doc_id_prefix=filename)
    return {"filename": filename, "chunks_added": num_chunks}
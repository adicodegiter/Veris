import os
import re
from pypdf import PdfReader
import chromadb
import ollama
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DISTANCE_THRESHOLD = 1.5
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)

if USE_GROQ:
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_MODEL = "openai/gpt-oss-20b"

client = chromadb.PersistentClient(path="./chroma_db")

STOP_WORDS = {"a", "am", "are", "do", "does", "i", "is", "know", "me", "my", "the", "what", "which", "with", "you"}

def _get_collection(session_id):
    """Each visitor gets their own isolated document collection, keyed by session_id."""
    return client.get_or_create_collection(name=f"session_{session_id}")

def _chat(messages):
    if USE_GROQ:
        response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages)
        return response.choices[0].message.content
    else:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        return response["message"]["content"]

def ingest_pdf(filepath, session_id, doc_id_prefix="doc"):
    collection = _get_collection(session_id)
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    full_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', full_text)

    chunks = _chunk_text(full_text)

    # Add the filename as a searchable chunk so questions about the document's
    # identity (e.g. "what week is this") can be answered using filename context,
    # not just the body text.
    filename_chunk = f"Document filename: {doc_id_prefix}"
    chunks = [filename_chunk] + chunks

    existing_count = collection.count()
    collection.add(
        documents=chunks,
        ids=[f"{doc_id_prefix}_{existing_count + i}" for i in range(len(chunks))]
    )
    return len(chunks)

def _chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = []
        char_count = 0
        i = start
        while i < len(words) and char_count < chunk_size:
            chunk_words.append(words[i])
            char_count += len(words[i]) + 1
            i += 1
        chunks.append(" ".join(chunk_words))
        overlap_chars = 0
        back = i
        while back > start and overlap_chars < overlap:
            back -= 1
            overlap_chars += len(words[back]) + 1
        start = back if back > start else i
    return chunks

def _validate_answer(answer, context):
    validation_prompt = f"""You are a strict fact-checker. Check whether the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER TO CHECK:
{answer}

Instructions:
- Check every claim in the ANSWER against the CONTEXT only.
- A claim is UNSUPPORTED if it's not directly stated or clearly implied by the CONTEXT.
- Respond in EXACTLY this format, nothing else:

VERDICT: [SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED]
UNSUPPORTED_CLAIMS: [list specific unsupported claims, or "None"]
REASONING: [one sentence]"""
    return _chat([{"role": "user", "content": validation_prompt}])

def ask_question(question, session_id):
    collection = _get_collection(session_id)

    if collection.count() == 0:
        return {"answer": "No documents have been uploaded yet. Upload a document to get started.", "sources": [], "validation": "N/A"}

    results = collection.query(
        query_texts=[question],
        n_results=collection.count(),
        include=["documents", "distances"]
    )
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]

    question_terms = {
        term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", question.lower())
        if term not in STOP_WORDS
    }

    filtered = [
        (chunk, dist) for chunk, dist in zip(retrieved_chunks, distances)
        if dist <= DISTANCE_THRESHOLD
        or any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", chunk.lower()) for term in question_terms)
    ]

    if not filtered:
        return {
            "answer": "I don't have relevant information in your documents to answer this.",
            "sources": [],
            "validation": "N/A (no context retrieved)"
        }

    context = "\n\n".join([c for c, d in filtered])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    answer_text = _chat([{"role": "user", "content": prompt}])

    if "don't know" in answer_text.lower() or "cannot" in answer_text.lower() or "no relevant" in answer_text.lower():
        validation = "SUPPORTED (model correctly declined to answer)"
    else:
        validation = _validate_answer(answer_text, context)

    return {
        "answer": answer_text,
        "sources": [c[:150] for c, d in filtered],
        "validation": validation
    }
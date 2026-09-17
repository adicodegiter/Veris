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
collection = client.get_or_create_collection(name="resume")

STOP_WORDS = {"a", "am", "are", "do", "does", "i", "is", "know", "me", "my", "the", "what", "which", "with", "you"}

def _chat(messages):
    """Unified chat function — uses Groq if an API key is set, otherwise falls back to local Ollama."""
    if USE_GROQ:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(model="llama3.2:3b", messages=messages)
        return response["message"]["content"]

def validate_answer(answer, context):
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

def ask_question(question):
    if collection.count() == 0:
        return {
            "answer": "I don't have any documents to answer from.",
            "validation": "VERDICT: N/A (no documents have been absorbed)",
            "sources": [],
        }

    results = collection.query(
        query_texts=[question],
        n_results=collection.count(),
        include=["documents", "distances"],
    )
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    question_terms = {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z+#.-]*", question.lower())
        if term not in STOP_WORDS
    }
    filtered = [
        (chunk, distance)
        for chunk, distance in zip(retrieved_chunks, distances)
        if distance <= DISTANCE_THRESHOLD
        or any(
            re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", chunk.lower())
            for term in question_terms
        )
    ]

    if not filtered:
        return {
            "answer": "I don't have relevant information in your documents to answer this.",
            "validation": "VERDICT: SUPPORTED (model correctly declined to answer)",
            "sources": [],
        }

    context = "\n\n".join(chunk for chunk, _ in filtered)
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    answer = _chat([{"role": "user", "content": prompt}])

    if "don't know" in answer.lower() or "cannot" in answer.lower() or "no relevant" in answer.lower():
        validation = "VERDICT: SUPPORTED (model correctly declined to answer)"
    else:
        validation = validate_answer(answer, context)

    return {
        "answer": answer,
        "validation": validation,
        "sources": [chunk for chunk, _ in filtered],
    }

def ingest_pdf(filepath, doc_id_prefix="doc"):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    full_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', full_text)

    chunks = _chunk_text(full_text)
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

        # Add whole words until we roughly hit chunk_size characters
        while i < len(words) and (
            char_count + len(words[i]) + (1 if chunk_words else 0) <= chunk_size
            or not chunk_words
        ):
            word = words[i]
            chunk_words.append(word)
            char_count += len(word) + (1 if len(chunk_words) > 1 else 0)
            i += 1

        chunks.append(" ".join(chunk_words))
        if i >= len(words):
            break

        overlap_chars = 0
        overlap_words = 0
        for word in reversed(chunk_words):
            overlap_chars += len(word) + (1 if overlap_words else 0)
            overlap_words += 1
            if overlap_chars >= overlap:
                break

        start = max(start + 1, i - overlap_words)

    return chunks
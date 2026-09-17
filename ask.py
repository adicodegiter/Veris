import re
from pypdf import PdfReader
import chromadb
import ollama

DISTANCE_THRESHOLD = 1.5

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

    result = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": validation_prompt}]
    )
    return result["message"]["content"]

# 1. Read the resume PDF
reader = PdfReader("resume.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# Clean up squished PDF text (e.g. "LanguagesPython" -> "Languages Python")
full_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', full_text)

# 2. Chunk it
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
print(f"Split resume into {len(chunks)} chunks")

# 3. Store in vector DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="resume")

if collection.count() == 0:
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Chunks added to vector database")

# 4. Ask a question
question = input("\nAsk a question about your resume: ")

# 5. Retrieve candidates using semantic similarity.
results = collection.query(
    query_texts=[question],
    n_results=collection.count(),
    include=["documents", "distances"]
)
retrieved_chunks = results["documents"][0]
distances = results["distances"][0]

# 6. Filter by relevance threshold
stop_words = {
    "a", "am", "are", "do", "does", "i", "is", "know", "me", "my",
    "the", "what", "which", "with", "you"
}
question_terms = {
    term
    for term in re.findall(r"[a-zA-Z][a-zA-Z+#.-]*", question.lower())
    if term not in stop_words
}

filtered = [
    (chunk, dist)
    for chunk, dist in zip(retrieved_chunks, distances)
    if dist <= DISTANCE_THRESHOLD
    or any(
        re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", chunk.lower())
        for term in question_terms
    )
]

print(f"\n--- Retrieved context (kept {len(filtered)}/{len(retrieved_chunks)} chunks above threshold) ---")
for i, (chunk, dist) in enumerate(filtered):
    print(f"[{i+1}] (distance: {dist:.3f}) {chunk[:150]}...")

# 7. Generate answer (or decline if nothing relevant)
if not filtered:
    answer_text = "I don't have relevant information in your documents to answer this."
    print("\n--- Answer ---")
    print(answer_text)
    print("\n--- Validation ---")
    print("VERDICT: N/A (no context retrieved, so no claim was made)")
else:
    context = "\n\n".join([c for c, d in filtered])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    answer_text = response["message"]["content"]

    print("\n--- Answer ---")
    print(answer_text)

    if "don't know" in answer_text.lower() or "cannot" in answer_text.lower() or "no relevant" in answer_text.lower():
        print("\n--- Validation ---")
        print("VERDICT: SUPPORTED (model correctly declined to answer)")
    else:
        print("\n--- Validation ---")
        verdict = validate_answer(answer_text, context)
        print(verdict)
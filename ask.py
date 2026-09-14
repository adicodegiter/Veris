from pypdf import PdfReader
import chromadb
import ollama

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
reader = PdfReader("resume.pdf")  # rename this to match your actual file name
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# 2. Chunk it into smaller pieces (simple fixed-size chunking for now)
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

# 3. Store chunks in a local vector database (ChromaDB handles embeddings automatically)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="resume")

# Only add if empty (avoids duplicating on every run)
if collection.count() == 0:
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Chunks added to vector database")

# 4. Ask a question
question = input("\nAsk a question about your resume: ")

# 5. Retrieve the most relevant chunks
results = collection.query(
    query_texts=[question],
    n_results=3,
    include=["documents", "distances"]
)
retrieved_chunks = results["documents"][0]
distances = results["distances"][0]
context = "\n\n".join(retrieved_chunks)

# 6. Generate an answer using the local LLM, grounded in retrieved chunks
print("\n--- Retrieved context (distance = lower is more relevant) ---")
for i, (chunk, dist) in enumerate(zip(retrieved_chunks, distances)):
    print(f"[{i+1}] (distance: {dist:.3f}) {chunk[:150]}...")
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

print("\n--- Validation ---")
verdict = validate_answer(answer_text, context)
print(verdict)
from pypdf import PdfReader

reader = PdfReader("resume.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
for i, c in enumerate(chunks):
    print(f"\n=== Chunk {i} ===")
    print(c)
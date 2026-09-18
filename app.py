import streamlit as st
import requests
import os

try:
    secret_api_url = st.secrets.get("API_URL")
except Exception:
    secret_api_url = None

API_URL = (os.getenv("API_URL") or secret_api_url or "http://127.0.0.1:8001").rstrip("/")

st.set_page_config(page_title="Veris", page_icon="🔍")
st.title("🔍 Veris")
st.caption("Ask questions about your documents — grounded, with sources, and hallucination checks.")

# --- Sidebar: document upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file is not None:
        if st.button("Absorb document"):
            with st.spinner("Reading and embedding document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/absorb", files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Added {result['chunks_added']} chunks from {result['filename']}")
                else:
                    st.error("Failed to absorb document.")

# --- Main: chat interface ---
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            response.raise_for_status()
            result = response.json()
            st.session_state.history.append({"question": question, "result": result})
        except requests.RequestException as error:
            st.error(f"The API request failed: {error}")
        except ValueError:
            st.error(f"The API returned an invalid response: {response.text[:500]}")

# Display conversation history, most recent first
for entry in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["result"]["answer"])

        verdict = entry["result"].get("validation", "")
        verdict_line = verdict.split("\n")[0] if "\n" in verdict else verdict
        is_unsupported = verdict_line.strip().upper().startswith("VERDICT: UNSUPPORTED") or verdict_line.strip().upper().startswith("VERDICT: PARTIALLY_SUPPORTED")

        if is_unsupported:
            st.warning(f"⚠️ Validation flagged this answer:\n\n{verdict}")
        else:
            st.info(f"✅ Validation: {verdict}")

        if entry["result"].get("sources"):
            with st.expander("View sources"):
                for i, src in enumerate(entry["result"]["sources"]):
                    st.text(f"[{i+1}] {src}")
import streamlit as st
import uuid
import rag_core

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Veris", page_icon="🔍")

session_id = st.session_state.setdefault("session_id", str(uuid.uuid4()))

st.title("🔍 Veris")
st.caption("Upload a document and ask questions — grounded, with sources, and hallucination checks.")
st.caption("Documents stay in temporary memory and clear after one hour of inactivity or an app restart.")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Add documents"):
        with st.spinner("Reading and embedding documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.size > MAX_UPLOAD_BYTES:
                    st.error(f"{uploaded_file.name} exceeds the 10 MB limit.")
                    continue
                try:
                    chunks_added = rag_core.ingest_document_bytes(
                        uploaded_file.getvalue(),
                        session_id,
                        doc_id_prefix=uploaded_file.name,
                    )
                    st.success(f"Added {uploaded_file.name} ({chunks_added} chunks)")
                except Exception as error:
                    st.error(f"Could not add {uploaded_file.name}: {error}")

    st.subheader("Current-session documents")
    saved_documents = rag_core.list_documents(session_id)
    if saved_documents:
        for filename in saved_documents:
            st.write(filename)
    else:
        st.caption("No documents uploaded yet.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.spinner("Thinking..."):
        try:
            result = rag_core.ask_question(question, session_id)
            st.session_state.history.append({"question": question, "result": result})
        except RuntimeError as error:
            st.error(str(error))

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
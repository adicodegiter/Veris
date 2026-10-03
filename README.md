# Veris

Veris lets a visitor upload documents and ask questions grounded in their contents.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository, branch `main`, and entry point `app.py`.
3. In the app settings, add this secret:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```

4. Deploy the app and share the generated `streamlit.app` URL.

The app uses in-memory ChromaDB collections. Documents are isolated per Streamlit session and are cleared after one hour without activity or when the app process restarts. Uploaded files are limited to 10 MB and are not written to disk. The LLM provider has its own free-tier quotas and availability limits.

For local development, install `requirements.txt`, set `GROQ_API_KEY` in `.env`, then run `streamlit run app.py`.

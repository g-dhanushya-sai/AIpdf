# AIpdf — Chat with multiple PDFs using Flask + LangChain + Gemini (Vertex AI)

This repository provides a minimal Flask web app that lets a user upload multiple PDF documents and chat with their combined content using LangChain and Google Vertex AI (Gemini) as the LLM.

Important notes
- You must configure Google Cloud credentials correctly. Vertex AI typically requires a service account JSON and `GOOGLE_APPLICATION_CREDENTIALS` or proper ADC (Application Default Credentials). An API key alone (the short `AIza...` key) may not be sufficient.
- Keep secrets out of source control. Use environment variables or a `.env` file (see `.env.example`).

Quick setup

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure credentials

- For Vertex AI (Gemini), create a service account with the required permissions and download the JSON key. Set `GOOGLE_APPLICATION_CREDENTIALS` to that file, or authenticate with `gcloud auth application-default login`.
- Or set `GOOGLE_API_KEY` in your environment but verify it works for your use-case.

3. Run the app

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=8080
```

What the app does
- Upload multiple PDFs and build a FAISS vector store of their chunks.
- Use LangChain's ConversationalRetrievalChain with Vertex AI (Gemini) as the LLM and the FAISS retriever.
- Simple web UI to ask questions and see answers + source documents.

Security & production
- Do not commit keys. Use environment variables or secret managers.
- For production, run behind a proper WSGI server (e.g. gunicorn) and enable HTTPS.

If something doesn't work (e.g., Vertex AI integration), you can switch to local embeddings (sentence-transformers) by leaving Google credentials unset; the app will fall back automatically for embeddings but LLM responses require a configured LLM.

License: choose one for your use.# AIpdf
# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask questions about it.

## Tech Stack
- **FastAPI** — backend API
- **ChromaDB** — vector store for document chunks
- **Sentence Transformers** — local embeddings (all-MiniLM-L6-v2)
- **Groq** — LLM inference (llama-3.3-70b-versatile)

## How It Works
1. Upload a PDF → it gets parsed, chunked into 500-word segments, embedded and stored in ChromaDB
2. Ask a question → the question is embedded, top-5 similar chunks are retrieved, sent to Groq LLM with the question
3. Answer comes strictly from the document context — no hallucination

## Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Set your `GROQ_API_KEY` in a `.env` file.
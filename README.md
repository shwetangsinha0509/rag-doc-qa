# RAG Document Q&A — LangChain Refactor

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask questions about it.

This is a refactored version of the [original raw implementation](https://github.com/shwetangsinha0509/rag-doc-qa/tree/main) which was built using low-level libraries directly. This version rebuilds the same pipeline using LangChain v1, demonstrating how the framework abstracts and replaces manual RAG components.

## Tech Stack
- **FastAPI** — backend API
- **ChromaDB** — vector store for document chunks
- **LangChain v1** — RAG pipeline orchestration
- **Sentence Transformers** — local embeddings via `langchain-huggingface` (all-MiniLM-L6-v2)
- **Groq** — LLM inference via `langchain-groq` (llama-3.3-70b-versatile)

## How It Works
1. Upload a PDF → `PyPDFLoader` parses it into `Document` objects (one per page, with metadata)
2. `RecursiveCharacterTextSplitter` chunks by paragraph → sentence → word boundaries, preserving page metadata in every chunk
3. Chunks are embedded and stored in ChromaDB via `Chroma.from_documents()`
4. Ask a question → `create_retrieval_chain` embeds the question, retrieves top-5 similar chunks, formats them into a `ChatPromptTemplate`, and calls the Groq LLM
5. Answer comes strictly from document context — no hallucination

## What Changed from the Raw Version

| Component | Raw version | LangChain version |
|---|---|---|
| PDF loading | Manual `PdfReader` loop → plain string | `PyPDFLoader` → `Document` objects with page metadata |
| Chunking | Custom word-based splitter | `RecursiveCharacterTextSplitter` — splits on paragraph/sentence/word boundaries |
| Embeddings | ChromaDB-native `SentenceTransformerEmbeddingFunction` | `HuggingFaceEmbeddings` — LangChain-native, works with any vector store |
| Vector storage | Manual `collection.add()` with ID generation | `Chroma.from_documents()` — one call handles embed + store |
| Retrieval + LLM | Manual query → string join → f-string prompt → raw API call | `create_retrieval_chain` + `create_stuff_documents_chain` — chain handles the full pipeline |

## Why LangChain
The raw version required manually wiring together every step — querying ChromaDB, building context strings, formatting prompts, calling the LLM. LangChain abstracts each of these into swappable components. Switching from ChromaDB to Pinecone, or from Groq to OpenAI, is now a one-line change.

## Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Set your `GROQ_API_KEY` in a `.env` file.
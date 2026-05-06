import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from rag.ingestor import ingest_pdf
from rag.retriever import query_document

load_dotenv()

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    collection_name: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Save uploaded PDF, ingest into ChromaDB, return collection name."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Unique collection per upload so multiple docs don't mix
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"

    save_path = UPLOAD_DIR / f"{collection_name}.pdf"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    chunk_count = ingest_pdf(str(save_path), collection_name)

    return {
        "message": "Document ingested successfully.",
        "collection_name": collection_name,
        "chunks": chunk_count
    }


@app.post("/query")
async def query(request: QueryRequest):
    """Take a question + collection name, return answer from document."""
    answer = query_document(request.question, request.collection_name)
    return {"answer": answer}


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
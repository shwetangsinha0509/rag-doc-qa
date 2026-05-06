import os
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

#ChromaDB client- stores data locally in a folder called chroma_store
chroma_client = chromadb.PersistentClient(path="chroma_store")

# Embedding model - runs locally, no api cost
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_or_create_collection(collection_name: str):
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

def chunk_text(text:str, chunk_size:int = 500, overlap:int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words=text.split()
    chunks=[]
    i=0
    while i<len(words):
        chunk=" ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap #overlap so context isnt lost at boundaries
    return chunks

def ingest_pdf(file_path: str, collection_name: str) -> int:
    """Parse PDF -> chunk -> embed -> store in ChromaDB. returns chunk count."""
    reader = PdfReader(file_path)

    full_text=""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)

    collection = get_or_create_collection(collection_name)

    # add chunks with unique ids
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return len(chunks)
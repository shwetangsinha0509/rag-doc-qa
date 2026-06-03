import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_store"

# Same model as before, now through LangChain's universal embedding interface
# Swapping to OpenAI/Cohere embeddings later = one line change
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def get_vectorstore(collection_name: str) -> Chroma:
    """Load an existing ChromaDB collection as a LangChain vectorstore."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )


def ingest_pdf(file_path: str, collection_name: str) -> int:
    """
    Load PDF → split → embed → store in ChromaDB.
    Returns chunk count.
    """

    # Step 1: Load — returns list of Document objects, one per page
    # Each Document has .page_content and .metadata (page number, source path)
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Split — smarter than our manual word splitter
    # Tries to split on: paragraphs → sentences → words → characters
    # chunk_size is in characters (1000) not words (500) — closer to how LLMs count tokens
    # Metadata from each Document is preserved in every chunk automatically
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Step 3: Embed + store — replaces manual collection.add() + ID generation
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return len(chunks)
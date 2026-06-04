import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rag.ingestor import get_vectorstore
from dotenv import load_dotenv

load_dotenv()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2
        )
    return _llm


def query_document(question: str, collection_name: str, top_k: int = 5) -> str:
    """Retrieve relevant chunks and answer the question using the modern LangChain chain."""

    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

Context:
{context}"""),
        ("human", "{input}")
    ])

    stuff_chain = create_stuff_documents_chain(
        llm=get_llm(),
        prompt=prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_chain
    )
    
    result = retrieval_chain.invoke({"input": question})

    return result["answer"]
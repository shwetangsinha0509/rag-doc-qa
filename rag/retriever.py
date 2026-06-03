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

    # Step 1: Load vectorstore and convert to retriever
    # get_vectorstore() returns a LangChain Chroma object (not raw ChromaDB)
    # .as_retriever() converts it into a BaseRetriever that the chain expects
    # search_kwargs={"k": top_k} is the LangChain way of saying n_results=5
    vectorstore = get_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Step 2: Define the prompt using ChatPromptTemplate
    # Modern LangChain uses chat-style messages instead of a single text block
    # "system" message — sets assistant behaviour and injects retrieved {context}
    # "human" message — carries the user's {input} question
    # The chain automatically fills {context} with retrieved chunks
    # and {input} with the user's question when invoked
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

Context:
{context}"""),
        ("human", "{input}")
    ])

    # Step 3: Build the chain in two explicit parts
    # This replaces the single RetrievalQA.from_chain_type() call from older LangChain
    # Splitting into two parts makes each step visible and independently testable

    # Part A — stuff_chain
    # "stuff" means: take all retrieved chunks, stuff them into {context}, call LLM
    # This is what your old code did manually:
    # context = "\n\n---\n\n".join(chunks) → prompt → client.chat.completions.create()
    stuff_chain = create_stuff_documents_chain(
        llm=get_llm(),
        prompt=prompt
    )

    # Part B — retrieval_chain
    # Wires the retriever to the stuff_chain
    # When invoked: embeds question → searches ChromaDB → passes chunks to stuff_chain
    # This replaces: collection.query(query_texts=[question], n_results=top_k)
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_chain
    )

    # Step 4: Run the chain
    # Old code: result = client.chat.completions.create(...) → result.choices[0].message.content
    # New code: one invoke call does retrieve + format + generate
    # Input key is "input" (matches the "human" message placeholder above)
    # Output key is "answer" (what create_retrieval_chain returns)
    result = retrieval_chain.invoke({"input": question})

    return result["answer"]
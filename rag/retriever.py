import os
from groq import Groq
from dotenv import load_dotenv
from rag.ingestor import get_or_create_collection

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def query_document(question: str, collection_name: str, top_k: int = 5) -> str:
    """Embed question → fetch top_k chunks → ask Groq → return answer."""

    collection = get_or_create_collection(collection_name)

    # Fetch the most relevant chunks from ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunks = results["documents"][0]  # list of top_k matching chunks

    if not chunks:
        return "I couldn't find relevant information in the document."

    # Build context block from retrieved chunks
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information in the document to answer this."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temp = more factual, less creative
    )

    return response.choices[0].message.content
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load GROQ_API_KEY from .env (the file you created earlier)
load_dotenv()

# Paths
INDEX_FOLDER = Path("indexes")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Same embedding model we used for indexing (must be identical)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load the three FAISS indexes we created in 01_indexing.py
print("Loading the three FAISS indexes...")
yc_db       = FAISS.load_local(str(INDEX_FOLDER / "yc_index"),      embeddings, allow_dangerous_deserialization=True)
wealth_db   = FAISS.load_local(str(INDEX_FOLDER / "wealth_index"),  embeddings, allow_dangerous_deserialization=True)
general_db  = FAISS.load_local(str(INDEX_FOLDER / "general_index"), embeddings, allow_dangerous_deserialization=True)
print("All indexes loaded")

# Initialize the language model – Groq is free tier, very fast, zero cost
print("Initializing LLM (Groq + Llama-3.2-3B)...")
llm = ChatGroq(
model="llama-3.1-8b-instant",      # stable, fast, free tier
    temperature=0.0,                # deterministic → perfect for RAG
    max_tokens=1024,
)

# Quick test so we know it works
test_response = llm.invoke("Say exactly: 'hello from Groq'")
print(f"LLM test: {test_response.content.strip()}")

# Simple keyword router – decides which vector store to use
def route_question(question: str):
    q = question.lower()
    if any(kw in q for kw in ["yc", "y combinator", "startup batch", "founder", "application"]):
        return "yc"
    elif any(kw in q for kw in ["wealth", "rich", "money", "economic", "inequality"]):
        return "wealth"
    else:
        return "general"

# Create three LangChain retrievers (they all use the same embedding model)
yc_retriever       = yc_db.as_retriever(search_kwargs={"k": 6})
wealth_retriever   = wealth_db.as_retriever(search_kwargs={"k": 6})
general_retriever  = general_db.as_retriever(search_kwargs={"k": 6})

# Dictionary to pick the right retriever at runtime
retrievers = {
    "yc":       yc_retriever,
    "wealth":   wealth_retriever,
    "general":  general_retriever,
}

# Final RAG prompt – tells the LLM to only use retrieved context
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert on Paul Graham's essays.
Answer the question using ONLY the following context. 
If the context does not contain enough information, say "I don't know from the available essays".

Context:
{context}

Question: {question}"""),
    ("human", "{question}")
])

# Helper to format retrieved documents nicely
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Full RAG chain – this is the complete 8-stage pipeline in one line
rag_chain = (
    # 1) Compute topic from the question
    RunnableLambda(
        lambda x: {
            "question": x["question"],
            "topic": route_question(x["question"]),
        }
    )
    # 2) Retrieve docs based on topic
    | RunnableLambda(
        lambda x: {
            "question": x["question"],
            "context": format_docs(
                retrievers[x["topic"]].invoke(x["question"])
            ),
        }
    )
    # 3) Build prompt, call LLM, parse to string
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\nFull RAG pipeline ready!")
print("="*60)

def ask_pg(question: str) -> str:
    """Ask a question about Paul Graham's essays using the RAG pipeline."""
    return rag_chain.invoke({"question": question})
def debug_retrieval(question: str):
    topic = route_question(question)
    retriever = retrievers[topic]
    docs = retriever.invoke(question)
    print(f"\n[DEBUG] Topic: {topic}")
    for i, d in enumerate(docs, start=1):
        print(f"\n--- Doc {i} ---")
        print(d.page_content[:500], "...")

DONT_KNOW_MSG = "I don't know from the available essays"

def answer_with_fallback(question: str) -> str:
    """Try RAG first. If RAG says it doesn't know, fall back to base LLM."""
    # 1) Use the RAG pipeline
    rag_answer = rag_chain.invoke({"question": question})
    
    # 2) If RAG couldn't answer, use general LLM knowledge
    if DONT_KNOW_MSG.lower() in rag_answer.lower():
        base_answer = llm.invoke(question).content
        return (
            f"{rag_answer}\n\n"
            f"(From general knowledge, not from the indexed essays: {base_answer})"
        )
    
    # 3) Otherwise, return the grounded RAG answer
    return rag_answer


# Test questions – try them one by one
test_questions = [
    "What does Paul Graham say about applying to Y Combinator?",
    "Why does Paul Graham think wealth is created by startups?",
    "What is Paul Graham's advice on writing essays?",
    "How did Paul Graham make money before YC?",
]
while True:
    q = input("\nAsk about Paul Graham (or 'exit'): ")
    if q.lower() in {"exit", "quit"}:
        break
    print("\nAnswer:", answer_with_fallback(q))

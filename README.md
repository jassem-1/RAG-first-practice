
A minimal working Retrieval-Augmented Generation (RAG) system that lets you ask questions about **all of Paul Graham’s essays** and get accurate, cited answers.
Goal: understand every step of a real Retrieval-Augmented Generation system by implementing it with small , clear and readable  code.

This is not a production app it is a learning demo you can run in 5 minutes and then extend yourself.
- No paid APIs (only Groq free tier for the LLM)
- Runs on any laptop
- Indexes are ~30 MB total
- Fully interactive – just type your question

## What it does right now (and what it doesn’t)

| Feature                        | Status   | Notes |
|--------------------------------|----------|-------|
| Loads all Paul Graham essays   | Done     | ~3 MB plain text |
| Splits into smart chunks       | Done     | 500-char chunks with overlap |
| Builds 3 separate vector stores (YC, Wealth, General) | Done | FAISS + all-MiniLM-L6-v2 |
| Routes your question to the right topic | Done | Simple keyword routing |
| Retrieves the 6 most similar chunks | Done | Fast vector search |
| Answers using Groq Llama-3.1-8B (free) | Done | Deterministic, no hallucinations |
| Interactive loop               | Done     | Just run and chat |
| Re-ranking / HyDE / multi-query | Not yet |  To add later |

## How to run 

git clone https://github.com/yourusername/pg-rag-chatbot.git
cd pg-rag-chatbot

python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

pip install langchain-community==0.3.0 langchain-groq sentence-transformers faiss-cpu python-dotenv tqdm

# Add your free Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 1. Build the indexes (run once)
python 01_indexing.py

# 2. Start chatting
python 02_pipeline.py

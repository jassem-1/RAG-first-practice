import os
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Folders
DATA_FOLDER = Path("data")
INDEX_FOLDER = Path("indexes")
INDEX_FOLDER.mkdir(exist_ok=True)

# Where our essays live
ESSAYS_PATH = DATA_FOLDER / "pg_essays.txt"

# Embedding model (lightweight, local, excellent quality/size ratio)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Load the full text file
print("Loading essays from disk...")
loader = TextLoader(str(ESSAYS_PATH), encoding="utf-8")
documents = loader.load()

# Split into smaller chunks (500 characters with 100-character overlap)
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # ~100–150 words per chunk – good for MiniLM
    chunk_overlap=100,       # overlap helps avoid cutting sentences in half
    length_function=len,
    add_start_index=True,    # saves character position in original file
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Initialize the embedding model (downloads ~80 MB the first time only)
print("Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},   # change to 'cuda' if you have a GPU
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model ready")

def add_topic_metadata(chunk: Document) -> Document:
    text = chunk.page_content.lower()
    
    if any(word in text for word in ["yc ", "y combinator", "startup batch", "founder", "application"]):
        chunk.metadata["topic"] = "yc"
    elif any(word in text for word in ["wealth", "rich", "money", "economic", "inequality"]):
        chunk.metadata["topic"] = "wealth"
    else:
        chunk.metadata["topic"] = "general"
        
    return chunk

# Apply metadata tagging to all chunks
print("Adding topic metadata to chunks...")
chunks_with_metadata = [add_topic_metadata(chunk) for chunk in tqdm(chunks)]

# Separate into three groups
yc_chunks = [c for c in chunks_with_metadata if c.metadata["topic"] == "yc"]
wealth_chunks = [c for c in chunks_with_metadata if c.metadata["topic"] == "wealth"]
general_chunks = [c for c in chunks_with_metadata if c.metadata["topic"] == "general"]

print(f"Topic distribution → YC: {len(yc_chunks)} | Wealth: {len(wealth_chunks)} | General: {len(general_chunks)}")


# Helper to create and save an index
def build_and_save_index(chunk_list, name):
    if not chunk_list:
        print(f"No chunks for {name} – skipping")
        return
    print(f"Building FAISS index for {name} ({len(chunk_list)} chunks)...")
    vectorstore = FAISS.from_documents(chunk_list, embeddings)
    index_path = INDEX_FOLDER / name
    vectorstore.save_local(str(index_path))
    print(f"→ Saved to indexes/{name}")

# Build all three
build_and_save_index(yc_chunks, "yc_index")
build_and_save_index(wealth_chunks, "wealth_index")
build_and_save_index(general_chunks, "general_index")

print("Indexing complete! All three vector stores are ready in the 'indexes/' folder")

if __name__ == "__main__":
    # Just re-running the whole file will rebuild everything (useful for experiments)
    pass
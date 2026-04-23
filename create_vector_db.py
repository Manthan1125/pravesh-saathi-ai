import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PDF_FOLDER = "pdfs"
TEXT_FOLDER = "knowledge"
PERSIST_DIR = "vector_db"

documents = []

# -------------------
# Load PDFs (ALL pages — these PDFs are already UIET-specific)
# -------------------

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        print(f"  Loading PDF: {file}")
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        docs = loader.load()

        for doc in docs:
            # Tag each page with its source file for better retrieval
            doc.metadata["source_file"] = file
            # Skip near-empty pages (headers/footers only)
            if len(doc.page_content.strip()) > 50:
                documents.append(doc)

        print(f"    -> {len(docs)} pages loaded")

# -------------------
# Load TXT (scraped website knowledge files)
# -------------------

for file in os.listdir(TEXT_FOLDER):
    if file.endswith(".txt"):
        print(f"  Loading TXT: {file}")
        loader = TextLoader(os.path.join(TEXT_FOLDER, file), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = file
            if len(doc.page_content.strip()) > 20:
                documents.append(doc)

print(f"\nTotal documents loaded: {len(documents)}")


# -------------------
# Smart Chunking
# -------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

split_docs = text_splitter.split_documents(documents)

print(f"Chunks created: {len(split_docs)}")


# -------------------
# Embeddings
# -------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------
# Generate IDs
# -------------------

ids = [str(uuid.uuid4()) for _ in range(len(split_docs))]


# -------------------
# Create Vector DB
# -------------------

vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    ids=ids,
    persist_directory=PERSIST_DIR
)

print("UIET Vector DB created successfully!")
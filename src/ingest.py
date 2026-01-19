import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# Path config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "scraper", "data", "clean_scraped_articles_english.json")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def ingest():
    print("         ---> Starting ingestion process...")
    # load data
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON file not found at {JSON_PATH}")
        return
    
    print(f"Loading data from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding = 'utf-8') as f:
        articles = json.load(f)

    documents = []

    for article in articles:
        source_url = article.get("url", "unknown")
        title = article.get("title", "No Title")
        content = article.get("content", "")

        page_content = f"Title: {title}\nSource: {source_url}\nContent: {content}"

        # Create Document object
        doc = Document(
            page_content=page_content,
            metadata={"source": source_url, "title": title}
        )
        documents.append(doc)

    # Chunking
    print("Chunking documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    # Embed and store
    print("Creating embeddings and storing in Chroma DB...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model, 
            persist_directory=DB_PATH
        )
    
    print(f"Ingestion complete. Vector DB stored at {DB_PATH}.")

if __name__ == "__main__":
    ingest()
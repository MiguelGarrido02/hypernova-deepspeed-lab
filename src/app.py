import streamlit as st
import requests
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Config
RUNPOD_API_URL = "http://213.192.2.124:40045/generate"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# Setting up (only run once)
st.set_page_config(page_title="Multiverse News RAG Assistant", layout="wide")

@st.cache_resource
def load_db():
    """
    Load the Vector Database and Embedding Model.
    We cache this so we don't reload the 80MB model on every click.
    """
    print("--- Loading Embedding Model & DB ---")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_PATH):
        st.error(f"Database not found at {DB_PATH}. Please run ingest.py first!")
        return None

    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return db

db = load_db()

# --- UI LAYOUT ---
st.title("🌌 Multiverse Computing - GenAI Ops Demo")
st.markdown("""
*Running on a Hybrid Cloud Architecture:*
- **Knowledge Base:** Local ChromaDB with LangChain & HuggingFace Embeddings
- **Inference Engine:** Remote GPU Cluster (2x RTX 4090 via DeepSpeed/vLLM)
""")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# logic loop
if prompt := st.chat_input("Ask about Multiverse technology..."):
    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# RAG
    with st.spinner("Searching internal knowledge base..."):
        if db:
            # Increase k to 5 to ensure we find distinct articles (we filter duplicates below)
            docs = db.max_marginal_relevance_search(prompt, k=5, fetch_k=20)
            
            context_entries = []
            seen_titles = set()
            
            for doc in docs:
                # Extract Metadata
                title = doc.metadata.get('title', 'No Title')
                source = doc.metadata.get('source', 'No Source')
                date = doc.metadata.get('date', 'No Date')
                
                # Deduplication: If we already have this article, skip it to get more variety
                if title in seen_titles:
                    continue
                
                seen_titles.add(title)
                
                # Stop if we have 3 distinct articles
                if len(context_entries) >= 3:
                    break
                
                # CLEANER FORMATTING: Explicitly tell the LLM what is Metadata
                entry = f"""
                --- ARTICLE {len(context_entries) + 1} ---
                Title: {title}
                Source: {source}
                Date: {date}
                Content Summary:
                {doc.page_content}
                ------------------------
                """
                context_entries.append(entry)
            
            context_text = "\n".join(context_entries)
        else:
            context_text = "No context available."

    # 3. Construct Messages List 
        system_instruction = f"""
        You are an expert assistant for Multiverse Computing.
        Answer the user's question using ONLY the provided Context below.
        If asked for news, list the Titles. Always cite the Source URL.
        CONTEXT:
        {context_text}
        """

        messages_payload = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

    # Call Remote Inference Server
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Querying Remote Tensor Parallel Model..."):
                payload = {
                    "messages": messages_payload,
                    "max_tokens": 512,
                    "temperature": 0.2
                }
                # Send HTTP Request GPUs
                response = requests.post(RUNPOD_API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    ai_answer = response.json().get("response", "Error: No response field")
                    message_placeholder.markdown(ai_answer)
                    
                    # Show what context was used 
                    # with st.expander("View Retrieved Context (Source Documents)"):
                    #     st.text(context_text)
                else:
                    ai_answer = f"Error from Inference Server: {response.text}"
                    message_placeholder.markdown(ai_answer)

        except Exception as e:
            ai_answer = f"Connection Failed:Is the RunPod server running?\nError: {e}"
            message_placeholder.markdown(ai_answer)

    # Save history
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
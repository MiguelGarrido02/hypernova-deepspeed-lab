import streamlit as st
import requests
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
RUNPOD_API_URL = "http://94.101.98.238:8000/generate"

# 2. PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# --- SETUP (RUNS ONCE) ---
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
- **Frontend & Retrieval:** Local Kubernetes (Minikube)
- **Inference Engine:** Remote GPU Cluster (2x RTX 4090 via DeepSpeed/vLLM)
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- THE LOGIC LOOP ---
if prompt := st.chat_input("Ask about Multiverse technology..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Retrieval (RAG)
    with st.spinner("Searching internal knowledge base..."):
        if db:
            # MMR Search for diversity (prevents duplicate 'About Us' chunks)
            docs = db.max_marginal_relevance_search(prompt, k=3, fetch_k=10)
            context_text = "\n\n".join([d.page_content for d in docs])
        else:
            context_text = "No context available."

    # 3. Construct Prompt for Llama 3 / HyperNova
    full_prompt = f"""
    [INST] You are an expert assistant for Multiverse Computing. 
    Use the Context below to answer the user's question accurately.
    
    Context:
    {context_text}
    
    Question: 
    {prompt}
    [/INST]
    """

    # Call Remote Inference Server
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("Querying Remote Tensor Parallel Model..."):
                payload = {
                    "prompt": full_prompt,
                    "max_tokens": 512,
                    "temperature": 0.7
                }
                # Send HTTP Request GPUs
                response = requests.post(RUNPOD_API_URL, json=payload, timeout=60)
                
                if response.status_code == 200:
                    ai_answer = response.json().get("response", "Error: No response field")
                    message_placeholder.markdown(ai_answer)
                    
                    # Show what context was used 
                    with st.expander("View Retrieved Context (Source Documents)"):
                        st.text(context_text)
                else:
                    ai_answer = f"Error from Inference Server: {response.text}"
                    message_placeholder.markdown(ai_answer)

        except Exception as e:
            ai_answer = f"Connection Failed:Is the RunPod server running?\nError: {e}"
            message_placeholder.markdown(ai_answer)

    # Save history
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
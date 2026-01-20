# 🌌 Multiverse Computing: GenAI Ops RAG Assistant

**A Hybrid Cloud Retrieval-Augmented Generation (RAG) System**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![vLLM](https://img.shields.io/badge/Inference-vLLM-green) ![LangChain](https://img.shields.io/badge/Orchestration-LangChain-orange) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![GPU](https://img.shields.io/badge/Hardware-2x%20RTX%203090%2F4090-purple)

This project demonstrates a production-grade **GenAI Operations (GenAI Ops)** architecture. It implements a RAG pipeline that scrapes Multiverse Computing's latest resources, processes them into a vector database, and serves answers via a distributed Large Language Model (HyperNova-60B) running on a remote GPU cluster.

---

## 🏗️ Architecture

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/340a8e6f-591b-44ef-82ce-209eb293eb74" />


The system is designed as a **Hybrid Cloud Architecture** to decouple the lightweight retrieval system from the heavy compute inference engine.

### Why this Architecture?

1.  **Distributed Inference (Tensor Parallelism):**
    * **The Problem:** The `HyperNova-60B` model is too large to fit into the VRAM of a single consumer/prosumer GPU (like an RTX 3090).
    * **The Solution:** We use **vLLM** with `tensor_parallel_size=2`. This splits the model layers across two GPUs simultaneously. The system uses Microscaling formats (`mxfp4`) and `bfloat16` for maximum efficiency.
2.  **Decoupled Frontend & Backend:**
    * **Local/CPU Node (`app.py`):** Handles the user interface, prompt engineering, and vector retrieval (ChromaDB). This is lightweight and cheap to run.
    * **Remote GPU Node (`server.py`):** A dedicated FastAPI server that does only one thing: token generation. This allows the GPU resources to be scaled independently of the user traffic.
3.  **Data Sovereignty & Freshness:**
    * Instead of relying on the model's frozen training data, we scrape fresh data from the source, cleaning and embedding it locally to ensure the model answers with up-to-date company information.

---

## 🧩 Components

### 1. Data Pipeline
* **`scraper.py`**: A robust web scraper using `requests` and `BeautifulSoup`. It handles pagination automatically to fetch all articles from the Multiverse Computing resources page.
* **`filtering_scraped_data.py`**: Pre-processing logic that detects language (filtering for English) and sanitizes footer/boilerplate text to prevent repetition in the context window.
* **`ingest.py`**: Chunks text using `RecursiveCharacterTextSplitter`, creates embeddings via `sentence-transformers/all-MiniLM-L6-v2`, and persists them into a **ChromaDB** vector store.

### 2. Inference Server (GPU)
* **`server.py`**: A FastAPI wrapper around the vLLM engine.
    * **Model:** `MultiverseComputingCAI/HyperNova-60B`
    * **Engine:** vLLM with PagedAttention and Tensor Parallelism.
    * **API:** Exposes a standard `/generate` endpoint expecting a chat template.

### 3. User Interface (Client)
* **`app.py`**: A Streamlit application.
    * Performs Semantic Search (MMR) on ChromaDB to find relevant news.
    * Injects retrieved metadata (Title, Date, Source) into the system prompt.
    * Sends the enriched prompt to the remote GPU server for final answer generation.

---

## 🚀 Installation & Usage

### Prerequisites
* **Client Side:** Python 3.10+, CPU sufficient.
* **Server Side:** Linux environment, **2x NVIDIA GPUs** (24GB VRAM each recommended), CUDA drivers installed.

### Step 1: Environment Setup
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_token
```

### Step 2: Data Pipeline (Run Locally)
First, gather the knowledge base.
```bash
# 1. Scrape data
python scraper.py

# 2. Clean and filter data
python filtering_scraped_data.py

# 3. Create Vector Database (ChromaDB)
python ingest.py
```

### Step 3: Start Inference Server (Run on GPU Cluster)
On your GPU machine (e.g., RunPod, AWS, localized cluster):
```bash
# Install server dependencies
pip install vllm fastapi uvicorn transformers

# Start the API (Ensure GPUs are visible)
python server.py
```
The server will initialize the model across both GPUs and listen on port 8000. 
Pod IP is hardcoded, remember to change it.

### Step 4: Run the UI (Run Locally)
Update RUNPOD_API_URL in app.py to point to your GPU server's IP address.
```bash
# Install client dependencies
pip install streamlit langchain langchain-community langchain-huggingface chromadb requests

# Launch the App
streamlit run app.py
```

---

## 📸 Screenshots
<img width="1907" height="893" alt="model suage prove" src="https://github.com/user-attachments/assets/f6a7dd6a-be4c-4233-97b6-9af837e59475" />
<img width="522" height="595" alt="parallelization proof" src="https://github.com/user-attachments/assets/a0ddb3ad-9838-45c4-adbf-da320c81dc50" />

---

## 🛠️ Tech Stack
- **Model Serving**: vLLM
- **LLM**: HyperNova-60B (via Hugging Face)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend)
- **Orchestration**: LangChain

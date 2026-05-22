# ⚛️ Hybrid RAG Physics Assistant

![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

A production-grade Retrieval-Augmented Generation (RAG) system built to accurately answer complex physics questions based on textbook data. 

This project implements an advanced **Hybrid Search** architecture (combining Semantic Vector Search with BM25 Keyword Search) and utilizes **Cohere Reranking** to achieve near-perfect retrieval accuracy. The entire system is automatically evaluated using the **RAGAS** framework.

---

## 🌟 Key Features

* **Hybrid Retrieval Pipeline**: Fuses exact keyword matching (BM25) with conceptual meaning (ChromaDB Vector Search) using Reciprocal Rank Fusion (RRF).
* **AI Reranking**: Uses Cohere's advanced reranking models to push the most relevant chunks to the very top.
* **Automated Evaluation Engine**: Includes a custom dataset generator and a RAGAS evaluation script to mathematically prove system accuracy (Faithfulness, Context Precision, Answer Relevancy).
* **Microservice Architecture**: Fully containerized using Docker Compose with isolated Frontend (Streamlit) and Backend (FastAPI) services.
* **Intelligent Ingestion**: Hashes document chunks to prevent duplicate data in the vector database upon re-ingestion.

## 🏗️ Architecture

1. **Frontend**: Streamlit UI (Port `8501`)
2. **Backend**: FastAPI (Port `8000`)
3. **Vector Database**: ChromaDB (Persistent local storage)
4. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Running locally)
5. **Generation LLM**: Groq (`llama-3.3-70b-versatile`)

## 🚀 Quick Start (Docker)

To run this project locally, you only need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hybrid-rag-assistant.git
   cd hybrid-rag-assistant
   ```

2. **Set up Environment Variables:**
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY and COHERE_API_KEY to the .env file
   ```

3. **Build and Run:**
   ```bash
   docker compose up --build -d
   ```
   *(Or simply run `make up` if you have Make installed).*

4. **Access the Application:**
   * **Chat UI (Streamlit)**: [http://localhost:8501](http://localhost:8501)
   * **API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 Evaluation Results

The system is rigorously tested against a Golden Dataset using RAGAS. 
**Latest Benchmark Scores (Llama-3.3-70B Judge):**
* **Faithfulness**: `0.8889` (No Hallucinations)
* **Answer Relevancy**: `0.9915` (Answers exactly what is asked)
* **Context Precision**: `0.9167` (Retrieves the correct needles from the haystack)

## 🛠️ Local Development (Without Docker)
If you wish to run the project without Docker, it uses the incredibly fast `uv` package manager:

```bash
uv sync --all-extras
# Start backend
uvicorn src.api.main:app --reload
# Start frontend
streamlit run frontend/app.py
```

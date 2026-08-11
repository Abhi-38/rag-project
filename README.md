# Medical RAG Assistant 🏥🤖

[![Python 3.11](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00.svg?style=flat)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama--Llama3-black.svg?style=flat)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--m3-yellow.svg?style=flat)](https://huggingface.co/BAAI/bge-m3)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> A production-grade Medical AI Assistant utilizing **Hybrid RAG** (Dense Embeddings + Sparse BM25), **Parent-Child Chunking**, **Reciprocal Rank Fusion (RRF)**, **Relevance Reranking**, and **Ollama (Llama 3)** for private, grounded, and accurate question answering with source citations from medical study documents.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [🐳 Docker Deployment (Recommended for End Users)](#-docker-deployment-recommended-for-end-users)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Configure Environment](#2-configure-environment)
  - [3. Place Medical Documents](#3-place-medical-documents)
  - [4. Start Application](#4-start-application)
  - [5. Access & Use Interface](#5-access--use-interface)
  - [6. Stopping & Restarting](#6-stopping--restarting)
  - [7. Docker Troubleshooting](#7-docker-troubleshooting)
- [⚙️ Local Conda / Python Setup (For Developers)](#️-local-conda--python-setup-for-developers)
- [Ingesting Documents](#-ingesting-documents)
- [API Documentation & Examples](#-api-documentation--examples)
- [Environment Configuration (.env)](#-environment-configuration-env)
- [Running Unit & Pipeline Tests](#-running-unit--pipeline-tests)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## ✨ Features

- 📄 **Smart Multi-Format Ingestion**: Parses PDFs using PyMuPDF (`fitz`) and DOCX files, extracting metadata including page numbers and section headings.
- 🧩 **Parent-Child Chunking**: Maintains large parent contexts (1000ch) for generation while retrieving smaller child chunks (250ch) for precise vector search matches.
- 🔍 **Hybrid Retrieval Pipeline**:
  - **Dense Search**: Semantic embeddings powered by `BAAI/bge-m3`.
  - **Sparse Search**: Lexical keyword matching using `rank-bm25`.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense & sparse rankings seamlessly.
  - **Relevance Reranking**: Term-density cross-scoring to prioritize top relevant contexts.
- 🦙 **100% Local & Private LLM**: Powered by local Ollama (`llama3:latest`) for zero data-leakage and zero external API costs.
- 💬 **Grounded Q&A & Conversational Greetings**: Answers medical questions *only* from indexed documents with source citations, while gracefully welcoming general conversational greetings.
- 🖥️ **Modern Glassmorphism Web Interface**: Dark mode interactive UI served directly by FastAPI at `http://localhost:8000`.

---

## 🏗️ Architecture Overview

```text
                    Docker Compose
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       FastAPI Application        Ollama
       (Port 8000)             (Port 11434)
              │                     │
              ├── Glassmorphic UI   └── Llama 3 LLM
              ├── BGE-M3
              ├── ChromaDB (Parent & Child)
              ├── BM25 Index
              └── RAG Pipeline
```

---

## 🐳 Docker Deployment (Recommended for End Users)

The Docker setup allows non-technical users to run the complete Medical RAG Assistant with a single command without installing Python, Conda, or manual dependencies.

### Prerequisites

Install **Docker Desktop**:
- [Download Docker Desktop for Windows / Mac / Linux](https://www.docker.com/products/docker-desktop/)

---

### 1. Clone Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/Abhi-38/rag-project.git
cd rag-project
```

---

### 2. Configure Environment

Copy `.env.example` to `.env`:

- **On macOS / Linux:**
  ```bash
  cp .env.example .env
  ```
- **On Windows (Command Prompt / PowerShell):**
  ```cmd
  copy .env.example .env
  ```

*(Note: Default settings in `.env.example` are pre-configured for Docker Compose deployment).*

---

### 3. Place Medical Documents

Place your medical study PDFs or DOCX files into the host machine directory:

```text
data/
└── sample_docs/
    ├── neurosurgery_textbook.pdf
    └── clinical_notes.docx
```

> 🔒 **Privacy Note**: Your private medical documents remain strictly on your local computer (`./data`) and are **NEVER** baked into the Docker image or committed to Git.

---

### 4. Start Application

Run Docker Compose to build and start all containers:

```bash
docker compose up
```

#### ℹ️ First Startup Note
On the very **first run**, Docker Compose will:
1. Build the FastAPI application image (~1 minute).
2. Download the `llama3:latest` model (~4.7 GB) into a persistent Docker volume.

> 💾 **Persistent Volume Storage**: Multi-GB LLM weights and vector database indexes are stored in persistent Docker volumes. Subsequent startups skip downloads and initialize in seconds!

---

### 5. Access & Use Interface

Once startup completes, open your web browser:

- **Web Application**: [http://localhost:8000](http://localhost:8000)
- **API Health Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **OpenAPI / Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

You can drag & drop new medical PDFs directly in the Web UI tab or click any suggested study question to test grounded retrieval.

---

### 6. Stopping & Restarting

- **To Stop the Application**:
  Press `Ctrl + C` in the terminal, or run:
  ```bash
  docker compose down
  ```

- **To Start Again**:
  ```bash
  docker compose up
  ```

> ⚠️ **Important**: Do **NOT** run `docker compose down -v` unless you explicitly intend to delete your persistent vector database and downloaded model weights!

---

### 7. Docker Troubleshooting

#### 1. Port 8000 is already in use
- **Cause**: Another service or server is using port 8000.
- **Fix**: Stop the existing application or edit `docker-compose.yml` to map port `8001:8000`.

#### 2. Insufficient Memory / Container Crash
- **Cause**: Running Llama 3 requires ~6 GB of RAM.
- **Fix**: Open **Docker Desktop Settings $\rightarrow$ Resources** and increase allocated RAM to at least 8 GB.

#### 3. View Container Logs
To inspect application or model logs:
```bash
# View app container logs
docker compose logs app

# View Ollama model container logs
docker compose logs ollama
```

---

## ⚙️ Local Conda / Python Setup (For Developers)

If you are developing locally using Conda/VS Code:

```bash
# 1. Create & activate Conda environment
conda create -n medvenv python=3.11 -y
conda activate medvenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure Ollama is running locally
ollama serve
ollama pull llama3:latest

# 4. Run application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📥 Ingesting Documents

You can ingest documents via:
1. **Web UI**: Upload via drag-and-drop in the **Document Ingestion** tab at `http://localhost:8000`.
2. **CLI Script**:
   ```bash
   python ingest_script.py "./data/sample_docs/your_medical_book.pdf"
   ```
3. **REST API**: `POST http://localhost:8000/api/ingest/upload`

---

## 🧪 Running Unit & Pipeline Tests

Run the complete test suite:

```bash
pytest -v
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.

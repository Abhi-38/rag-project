# Medical RAG Assistant 🏥🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00.svg?style=flat)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama--Llama3.1-black.svg?style=flat)](https://ollama.ai/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--m3-yellow.svg?style=flat)](https://huggingface.co/BAAI/bge-m3)

> A production-grade Medical AI Assistant utilizing **Hybrid RAG** (Dense Embeddings + Sparse BM25), **Parent-Child Chunking**, **Reciprocal Rank Fusion (RRF)**, **FlashRank Reranking**, and **Ollama (Llama 3.1)** for private, grounded, and accurate question answering with source citations from medical study documents.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Prerequisites](#-prerequisites)
- [Quick Start & Installation](#-quick-start--installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Install & Run Ollama](#4-install--run-ollama)
  - [5. Configure Environment Variables](#5-configure-environment-variables)
- [Ingesting Documents](#-ingesting-documents)
  - [Option A: Command Line Script](#option-a-command-line-script)
  - [Option B: Via REST API](#option-b-via-rest-api)
- [Running the Application](#-running-the-application)
- [API Documentation & Examples](#-api-documentation--examples)
- [Environment Configuration (.env)](#-environment-configuration-env)
- [Running Unit & Pipeline Tests](#-running-unit--pipeline-tests)
- [Project Directory Structure](#-project-directory-structure)
- [Troubleshooting & FAQ](#-troubleshooting--faq)
- [License](#-license)

---

## ✨ Features

- 📄 **Smart Multi-Format Ingestion**: Parses PDFs using PyMuPDF (`fitz`), automatically extracting metadata including page numbers and section headings.
- 🧩 **Parent-Child Chunking**: Maintains large parent contexts for generation while retrieving smaller child chunks for precise vector search matches.
- 🔍 **Hybrid Retrieval Pipeline**:
  - **Dense Search**: Semantic embeddings powered by `BAAI/bge-m3`.
  - **Sparse Search**: Lexical keyword matching using `rank-bm25`.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense & sparse rankings seamlessly.
  - **FlashRank Reranking**: Ultra-fast cross-encoder reranking to prioritize top-k relevant contexts.
- 🦙 **100% Local & Private LLM**: Powered by local Ollama (`llama3.1`) for zero data-leakage and zero API costs.
- 💬 **Grounded Q&A with Source Citations**: Synthesizes responses based *only* on context and provides explicit source metadata (PDF file, page number, section heading).
- ⚡ **Production FastAPI Server**: Async endpoints, full input validation with Pydantic v2, CORS support, and automatic OpenAPI/Swagger UI docs.

---

## 🏗️ Architecture Overview

```text
               +-----------------------------------+
               |  Medical PDF Study Documents      |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------+-----------------+
               | PyMuPDF Parsing & Page Extraction |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------+-----------------+
               |      Parent-Child Chunking        |
               | (Parent: 1000ch, Child: 250ch)   |
               +-----------------+-----------------+
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
+-------+-----------------------+       +-----------------+---------------------+
| Dense Embeddings (bge-m3)     |       | Sparse Lexical Index (BM25)         |
| Stored in ChromaDB            |       | Keyword matching                    |
+-------+-----------------------+       +-----------------+---------------------+
        |                                                 |
        +------------------------+------------------------+
                                 |
                                 v
               +-----------------+-----------------+
               | Reciprocal Rank Fusion (RRF)     |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------+-----------------+
               |      FlashRank Reranking          |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------+-----------------+
               |   Context-Grounded LLM Prompt    |
               |    (Ollama / Llama 3.1)           |
               +-----------------+-----------------+
                                 |
                                 v
               +-----------------+-----------------+
               |  Grounded Response + Citations    |
               +-----------------------------------+
```

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.10+** (Python 3.9 - 3.12 supported)
- **Git**
- **Ollama** (Required to serve local LLMs like Llama 3.1) -> Download from [ollama.ai](https://ollama.ai/)

---

## 🚀 Quick Start & Installation

Follow these steps to set up the project locally on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-project.git
cd rag-project
```

### 2. Create a Virtual Environment

It is strongly recommended to use a Python virtual environment to manage dependencies:

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **On Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

Upgrade `pip` and install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install & Run Ollama

1. **Install Ollama**: Follow instructions at [ollama.ai](https://ollama.ai).
2. **Start the Ollama Service** (if not running in the background):
   ```bash
   ollama serve
   ```
3. **Pull the Llama 3.1 Model**:
   ```bash
   ollama pull llama3.1
   ```
   *(Note: You can also use other models like `llama3`, `mistral`, or `gemma` by updating `.env`)*

### 5. Configure Environment Variables

Copy the provided template `.env.example` to `.env`:

- **On Linux / macOS:**
  ```bash
  cp .env.example .env
  ```
- **On Windows:**
  ```cmd
  copy .env.example .env
  ```

Default configuration in `.env`:
```env
APP_NAME=MedicalRAGAssistant
ENV=development

LLM_MODEL=llama3.1
LLM_TEMPERATURE=0.0
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_MODEL=BAAI/bge-m3
VECTOR_DB=chromadb
VECTOR_DB_DIR=./vectorstore

PARENT_CHUNK_SIZE=1000
PARENT_CHUNK_OVERLAP=150
CHILD_CHUNK_SIZE=250
CHILD_CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=15
TOP_K_RERANKED=5
```

---

## 📥 Ingesting Documents

You must ingest medical documents (PDFs) into ChromaDB before querying the system.

### Option A: Command Line Script

Run the ingestion script pointing to a sample or custom PDF file:

```bash
# Ingest default sample document or custom PDF
python ingest_script.py "./data/sample_docs/Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf"
```

Output should show:
```text
============================================================
Starting Production RAG Ingestion Pipeline
Document: Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery_by_Ravi_Ramamurthi.pdf
============================================================

[Step 1/3] Parsing document and splitting into semantic chunks...
[OK] Generated X semantic chunks with preserved metadata.

[Step 2/3] Generating dense embeddings using BGE model...
[OK] Generated X normalized embedding vectors.

[Step 3/3] Indexing chunks & embeddings into ChromaDB vector database...

============================================================
INGESTION SUCCESSFUL
============================================================
```

### Option B: Via REST API

You can also trigger ingestion programmatically via HTTP (see API section below).

---

## 💻 Running the Application

Start the FastAPI development server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Alternatively, run directly with python:
```bash
python -m app.main
```

Once started, the API will be available at:
- **API Root**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs (OpenAPI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Documentation & Examples

### 1. Medical Query Endpoint (`POST /api/query`)

Submits a user question to the Hybrid RAG pipeline and returns a grounded answer with document sources.

- **URL**: `http://localhost:8000/api/query`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

**Sample Request Body:**
```json
{
  "query": "What are the primary clinical indications for surgical intervention in intracerebral hemorrhage?"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the primary clinical indications for surgical intervention in intracerebral hemorrhage?"}'
```

**Sample Response:**
```json
{
  "query": "What are the primary clinical indications for surgical intervention in intracerebral hemorrhage?",
  "answer": "Based on the provided medical text, surgical intervention for intracerebral hemorrhage is indicated when...",
  "sources": [
    {
      "source": "Ramamurthi_and_Tandon’s_Textbook_of_Neurosurgery.pdf",
      "page": 42,
      "heading": "Chapter 4: Intracerebral Hemorrhage Management"
    }
  ],
  "grounded": true,
  "retrieved_contexts": [...]
}
```

---

### 2. Document Ingestion Endpoint (`POST /api/ingest`)

Triggers document parsing, chunking, embedding, and indexing into ChromaDB.

- **URL**: `http://localhost:8000/api/ingest`
- **Method**: `POST`

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./data/sample_docs/your_medical_book.pdf"}'
```

---

### 3. Direct LLM Prompt Endpoint (`POST /generate`)

Generates a response directly from the configured Ollama LLM without retrieval.

- **URL**: `http://localhost:8000/generate`
- **Method**: `POST`

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Define Glasgow Coma Scale (GCS)."}'
```

---

## ⚙️ Environment Configuration (.env)

All application settings are managed cleanly via environment variables using Pydantic Settings.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `MedicalRAGAssistant` | Name of the FastAPI application. |
| `ENV` | `development` | Environment mode (`development` or `production`). |
| `LLM_MODEL` | `llama3.1` | Local Ollama model name to use for generation. |
| `LLM_TEMPERATURE` | `0.0` | Temperature setting for LLM responses (0.0 for deterministic answers). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL where the Ollama service is listening. |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Hugging Face model for generating dense embeddings. |
| `VECTOR_DB` | `chromadb` | Vector database backend. |
| `VECTOR_DB_DIR` | `./vectorstore` | Local persistent directory for ChromaDB storage. |
| `PARENT_CHUNK_SIZE` | `1000` | Character size for parent context chunks. |
| `PARENT_CHUNK_OVERLAP` | `150` | Overlap size between parent chunks. |
| `CHILD_CHUNK_SIZE` | `250` | Character size for child retrieval chunks. |
| `CHILD_CHUNK_OVERLAP` | `50` | Overlap size between child chunks. |
| `TOP_K_RETRIEVAL` | `15` | Number of top candidates to retrieve before reranking. |
| `TOP_K_RERANKED` | `5` | Final top-k chunks passed to the LLM context. |

---

## 🧪 Running Unit & Pipeline Tests

The repository includes test suites using `pytest` to test ingestion, chunking, vector store indexing, and retrieval pipeline components.

Run all unit tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

---

## 📁 Project Directory Structure

```text
rag-project/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── ingest.py        # /api/ingest endpoint
│   │       ├── query.py         # /api/query endpoint
│   │       └── llm_routes.py    # /generate direct LLM endpoint
│   ├── core/
│   │   └── config.py            # Pydantic Settings & environment variables
│   ├── ingestion/
│   │   ├── embedder.py          # Embedding generation service
│   │   ├── parser.py            # PDF document parser & metadata extractor
│   │   └── loaders/
│   │       └── pipeline.py      # Ingestion pipeline orchestrator
│   ├── services/
│   │   ├── embedding_service.py # Core sentence-transformers embedder
│   │   ├── hybrid_retriever.py  # BM25 + Vector + RRF + FlashRank retriever
│   │   ├── llm_service.py       # Ollama API client integration
│   │   ├── rag_chain.py         # Grounded response builder & citation logic
│   │   └── vector_store.py      # ChromaDB integration wrapper
│   └── main.py                  # FastAPI application entry point
├── data/
│   └── sample_docs/             # Storage folder for medical PDFs
├── tests/                       # Pytest automated test files
├── vectorstore/                 # Local ChromaDB persistent database directory
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── ingest_script.py             # CLI script for document ingestion
├── requirements.txt             # Python dependencies manifest
└── README.md                    # Project documentation
```

---

## ❓ Troubleshooting & FAQ

### 1. `ConnectionRefusedError` / `Failed to connect to localhost:11434`
- **Cause**: Ollama service is not running locally.
- **Fix**: Open a terminal and run `ollama serve`. Make sure Ollama is listening on port `11434`.

### 2. `model 'llama3.1' not found`
- **Cause**: The specified Ollama model hasn't been pulled yet.
- **Fix**: Run `ollama pull llama3.1` in your terminal. Alternatively, change `LLM_MODEL` in `.env` to a model you already have downloaded (e.g. `llama3` or `mistral`).

### 3. First ingestion or query runs slowly
- **Cause**: On the initial run, Hugging Face automatically downloads the `BAAI/bge-m3` embedding model weights (~2.2 GB) and `FlashRank` model files.
- **Fix**: Subsequent runs will use locally cached weights and will execute instantly.

### 4. ChromaDB persistence issues on Windows
- **Cause**: SQLite/ChromaDB file permission or lock issues.
- **Fix**: Ensure `vectorstore` folder has write permissions and no other python process is locking the database.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.

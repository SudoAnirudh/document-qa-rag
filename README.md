# Document Q&A RAG Service (`document-qa-rag`)

A production-minded, lightweight, well-structured Document Q&A Retrieval-Augmented Generation (RAG) backend service built with Python 3.12, FastAPI, ChromaDB, and NVIDIA NIM AI Models.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    Client([Client Application]) -->|POST /upload PDF/TXT| UploadRoute[Upload Endpoint]
    Client -->|POST /query question| QueryRoute[Query Endpoint]

    subgraph Ingestion Pipeline
        UploadRoute --> Parser[IngestionService / PyPDF & UTF-8]
        Parser --> Chunker[ChunkingService / RecursiveTextSplitter]
        Chunker --> Embedder[EmbeddingService / nvidia/llama-3.2-nv-embedqa-1b-v2]
        Embedder --> VectorDB[(VectorStoreService / ChromaDB Persistent)]
    end

    subgraph Retrieval & Generation Pipeline
        QueryRoute --> Retriever[RetrievalService / Semantic Search]
        Retriever --> VectorDB
        Retriever --> ThresholdEvaluator{Similarity >= 0.35?}
        ThresholdEvaluator -->|No| Reject[Grounding Decision: FALSE\n"not found in the provided documents"]
        ThresholdEvaluator -->|Yes| Generator[GenerationService / meta/llama-3.2-11b-vision-instruct]
        Generator --> Accept[Grounding Decision: TRUE\nGrounded Answer + Sources]
    end
```

---

## 🛠 Tech Stack & Core Dependencies

- **Language**: Python 3.12
- **Framework**: FastAPI + Pydantic v2
- **Parser**: `pypdf` (PDF extraction) & UTF-8 text decoder
- **Chunker**: `RecursiveCharacterTextSplitter` from `langchain-text-splitters` (used strictly as an isolated text processing utility)
- **Vector Database**: `chromadb` (`PersistentClient` with cosine metric space)
- **Embeddings**: NVIDIA NIM API `nvidia/llama-3.2-nv-embedqa-1b-v2` (with automatic local ONNX fallback)
- **LLM Generator**: NVIDIA NIM API `meta/llama-3.2-11b-vision-instruct` (system prompt strictly forcing document grounding)
- **API Interface**: OpenAI-compatible client endpoint (`https://integrate.api.nvidia.com/v1`)
- **Testing**: `pytest` + `httpx` (51 unit and integration tests passing)

---

## ⚡ Quickstart & Setup

### 1. Requirements & Environment
Ensure Python 3.12+ is installed.

```bash
git clone <repository_url>
cd document-qa-rag

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file in the project root:

```ini
# NVIDIA NIM API Credentials (OpenAI-compatible)
OPENAI_API_KEY=nvapi-your-nvidia-api-key
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_EMBEDDING_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
OPENAI_GENERATION_MODEL=meta/llama-3.2-11b-vision-instruct

# Path Configurations
CHROMA_PATH=./chroma_db
DATA_PATH=./data

# RAG System Default Parameters
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.35
```

### 3. Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

Access API Docs (Swagger UI): `http://localhost:8002/docs`

---

## 🧪 Running Tests & Evaluation

### Run Test Suite (51 Unit & Integration Tests)
```bash
pytest -v
```

### Run Empirical Experiments Script
```bash
python -m evaluation.evaluate
```

---

## 📋 API Endpoints

### 1. `GET /health`
Health check endpoint verifying API availability.

**Response (200 OK):**
```json
{
  "status": "ok"
}
```

---

### 2. `POST /upload`
Upload and ingest a PDF or TXT document into the vector store.

**Request:** `multipart/form-data` with `file` field.

**Response (200 OK):**
```json
{
  "document_id": "doc_a1b2c3d4e5f6",
  "filename": "sample_tech_guide.pdf",
  "chunk_count": 19,
  "message": "Document successfully ingested and indexed."
}
```

---

### 3. `POST /query`
Execute semantic search, apply similarity threshold evaluation, and return a grounded answer.

**Request Body:**
```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "document_id": "doc_a1b2c3d4e5f6",
  "top_k": 5
}
```

**Response (Grounded - 200 OK):**
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation to answer questions based on external context.",
  "sources": [
    {
      "document_id": "doc_a1b2c3d4e5f6",
      "chunk_index": 0,
      "text": "Retrieval-Augmented Generation (RAG) enhances LLM responses...",
      "score": 0.82
    }
  ],
  "grounded": true
}
```

**Response (Ungrounded / Hallucination Shield - 200 OK):**
```json
{
  "answer": "not found in the provided documents",
  "sources": [],
  "grounded": false
}
```

---

## 📊 Empirical Experiment Summaries

1. **Chunking Strategy Comparison**:
   - `Config B (500/50)` selected as optimal default (19 chunks, avg length 345 chars), balancing context preservation with retrieval precision.
2. **Top-K Retrieval Evaluation**:
   - `top_k=5` selected (1.67x bloat factor, ~265 tokens), offering optimal context coverage without prompt bloat.
3. **Grounding Threshold Selection**:
   - `threshold=0.35` selected: In-doc scores (0.75 - 0.82) trigger LLM generation; unsupported scores (0.05 - 0.28) trigger hallucination shield rejection.

---

## ⚠️ Known Limitations & Future Work

- **Single Node Storage**: ChromaDB is configured with disk persistence (`PersistentClient`); multi-node clustering requires an external vector database setup.
- **Synchronous Ingestion**: Processing large PDFs (>50 MB) runs synchronously on API upload routes; background queue processing (e.g. ARQ / TaskIQ) can be added for high throughput.

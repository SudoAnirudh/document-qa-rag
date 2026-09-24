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
        Chunker --> Embedder[EmbeddingService / OpenAI / Local ONNX Fallback]
        Embedder --> VectorDB[(VectorStoreService / ChromaDB Persistent)]
    end

    subgraph Retrieval & Generation Pipeline
        QueryRoute --> Retriever[RetrievalService / Semantic Search]
        Retriever --> VectorDB
        Retriever --> ThresholdEvaluator{Similarity >= 0.50?}
        ThresholdEvaluator -->|No| Reject[Grounding Decision: FALSE\n"not found in the provided documents"]
        ThresholdEvaluator -->|Yes| Generator[GenerationService / LLM Generation]
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
- **Embeddings**: `text-embedding-3-small` / OpenAI API (with automatic local ONNX fallback via `all-MiniLM-L6-v2`)
- **LLM Generator**: OpenAI / NVIDIA NIM API (system prompt strictly forcing document grounding)
- **Testing**: `pytest` + `httpx` (52 unit and integration tests passing)

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
# API Credentials
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-4o-mini

# Path Configurations
CHROMA_PATH=./chroma_db
DATA_PATH=./data

# RAG System Default Parameters
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEFAULT_TOP_K=5
DEFAULT_SIMILARITY_THRESHOLD=0.50
```

### 3. Running the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

Access API Docs (Swagger UI): `http://localhost:8002/docs`

---

## 🧪 Running Tests & Real Retrieval Evaluation

### Run Test Suite (52 Unit & Integration Tests)
```bash
.venv/bin/pytest -v
```

### Run Real Vector Retrieval Evaluation & Benchmark Suite
```bash
PYTHONPATH=. .venv/bin/python evaluation/evaluate.py
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
      "score": 0.6635
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

## 📊 Empirical Experiment & Verification Results

All evaluation scripts run **REAL vector embedding and ChromaDB retrieval** against `test_document.txt` using the full application pipeline.

1. **Chunking Strategy Comparison**:
   - `Config B (500/50)` selected as optimal default (19 chunks, avg length 345 chars), balancing context preservation with retrieval precision.
2. **Top-K Retrieval Evaluation**:
   - `top_k=5` selected (avg context ~1,852 chars, ~463 estimated tokens), offering optimal context coverage without prompt bloat.
3. **Similarity Threshold & Leak Rate Verification**:
   - **Real In-Document Similarity Scores** (`q1`–`q4`): `0.5177` – `0.7485` (Mean = 0.6268)
   - **Real Related-Unsupported Similarity Scores** (`q5`–`q6`): `0.2821` – `0.4456`
   - **Real Unrelated Questions** (`q7`–`q8`): `0.0519` – `0.1086`
   - **Calibrated Gate Threshold**: Set to `0.50` (or `0.48`), resulting in:
     - **In-Document Hit Rate**: **100.0% (4/4)**
     - **Unsupported/Unrelated Leak Rate**: **0.0% (0/4)** (prevents related-but-unsupported questions like `q5` @ 0.4456 from triggering LLM generation).

---

## ⚠️ Known Limitations & Future Work

- **Single Node Storage**: ChromaDB is configured with disk persistence (`PersistentClient`); multi-node clustering requires an external vector database setup.
- **Synchronous Ingestion**: Processing large PDFs (>50 MB) runs synchronously on API upload routes; background queue processing (e.g. ARQ / TaskIQ) can be added for high throughput.

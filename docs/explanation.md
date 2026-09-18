# RAG System Architecture & Empirical Design Explanation

---

## 1. Design Parameters Chosen

The parameter configuration of `document-qa-rag` was selected based on empirical experiments executed on technical documentation:

### a. Chunking Configuration (`chunk_size=500`, `chunk_overlap=50`)
- **Config A (300/30)**: Produced 42 chunks with an average length of 159.93 characters. The small chunk size fragmented conceptual paragraphs across boundaries, degrading semantic retrieval relevance.
- **Config B (500/50)**: Produced 19 chunks with an average length of 345.05 characters. This provided the optimal trade-off: each chunk preserves a complete conceptual thought while keeping vector index memory footprint low.
- **Config C (800/80)**: Produced 12 chunks with an average length of 538.83 characters. While context per chunk was higher, broader chunks introduced noise into vector similarity matching.
- **Decision**: `chunk_size=500` and `chunk_overlap=50` (Config B) selected as system default.

### b. Top-K Retrieval (`top_k=5`)
- **top_k=3**: Retrieved 3 chunks (~148 tokens, 1.0x baseline bloat factor). Sufficient for simple factual queries, but missed supporting detail for multi-part questions.
- **top_k=5**: Retrieved 5 chunks (~265 tokens, 1.67x bloat factor). Provided optimal recall of supporting context without exceeding target prompt limits.
- **top_k=8**: Retrieved 8 chunks (~531 tokens, 2.67x bloat factor). Caused significant prompt bloat with diminishing marginal context gains.
- **Decision**: `top_k=5` locked as optimal default.

### c. Similarity Threshold (`threshold=0.35`)
- ChromaDB cosine distance metric produces distance scores in `[0, 2]`, which map to similarity via `similarity = 1.0 - distance`.
- **In-document relevant queries**: Empirical similarity scores ranged between `0.75` and `0.82` (e.g. `0.82` for RAG definition, `0.78` for embedding model name).
- **Related unsupported & unrelated queries**: Similarity scores ranged between `0.05` and `0.28` (e.g. `0.28` for RAM configuration on Render, `0.08` for capital of Australia).
- **Decision**: `threshold=0.35` acts as a strict halluncination shield, routing low-similarity queries directly to `grounded=False` without calling the LLM generator.

---

## 2. Observed Failures During Testing

During system implementation and testing, several edge-case failures were empirically observed and remediated:

1. **Corrupt / Empty PDF Text Extraction**:
   - *Failure*: When uploading corrupt or image-only PDF files, `pypdf` returned empty string content (`""`), causing downstream chunking to fail or attempt zero-chunk indexing.
   - *Fix*: Added validation in `IngestionService` raising `InvalidFileError("Extracted document text is empty")` mapped to `HTTP 400 Bad Request`.

2. **Cos-Distance to Similarity Metric Mapping**:
   - *Failure*: Initial queries against Chroma vector store returned cosine distance values (`0.18` for close match), which when compared directly against a similarity threshold caused false rejections.
   - *Fix*: Standardized score normalization `score = max(0.0, min(1.0, 1.0 - raw_distance))` across `VectorStoreService` to guarantee normalized similarity in `[0.0, 1.0]`.

3. **External Service Resiliency**:
   - *Failure*: External OpenAI API timeouts or rate-limits caused unhandled runtime exceptions resulting in `HTTP 500` errors.
   - *Fix*: Wrapped embedding and generation calls with explicit `try...except` blocks mapping `APITimeoutError` and `RateLimitError` to `ExternalServiceError` (`HTTP 502 Bad Gateway`).

---

## 3. Metrics Tracked

The service records structured, machine-readable performance metrics in logger streams:

- **`retrieval_ms`**: Latency for query embedding generation + Chroma vector store similarity lookup.
- **`generation_ms`**: Latency for OpenAI `gpt-4o-mini` grounded completion (or `0.00` when grounding threshold rejects).
- **`total_ms`**: End-to-end HTTP request processing duration.
- **Prompt Bloat Factor**: Ratio of context tokens delivered to the LLM relative to baseline `top_k=3` (`1.0x`, `1.67x`, `2.67x`).

Example Structured Log Entry:
```text
[INFO] query_metric embedding_ms=42.10 retrieval_ms=85.30 generation_ms=310.40 total_ms=438.20 grounded=true
```

---

## 4. Unfinished / Next Steps

To transition this lightweight RAG service into an enterprise production deployment, the following enhancements are recommended:

1. **Asynchronous Task Queue for Ingestion**:
   - Offload heavy PDF parsing and embedding generation to background workers (e.g. Celery / TaskIQ) for non-blocking document uploads.

2. **Cross-Encoder Re-Ranking**:
   - Implement a secondary re-ranking stage (e.g. `FlashRank` or `Cohere Rerank`) after initial vector retrieval to refine chunk ordering before prompt assembly.

3. **Hybrid Search (BM25 + Dense Vectors)**:
   - Combine sparse BM25 keyword matching with dense vector embeddings via Reciprocal Rank Fusion (RRF) to improve retrieval on exact keyword / code identifier queries.

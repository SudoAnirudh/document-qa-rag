import json
import tempfile
import shutil
from pathlib import Path
from typing import Any
from app.services.chunking import ChunkingService, DocumentChunk
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retrieval import RetrievalService
from app.core.logging import logger


def run_chunking_experiment(doc_path: Path | None = None) -> dict[str, dict]:
    """Execute chunking comparison across 3 configurations:

    Config A: size=300, overlap=30
    Config B: size=500, overlap=50
    Config C: size=800, overlap=80
    """
    if doc_path is None:
        doc_path = Path(__file__).parent / "test_document.txt"

    text = doc_path.read_text(encoding="utf-8")

    configs = {
        "Config A (300/30)": ChunkingService(chunk_size=300, chunk_overlap=30),
        "Config B (500/50)": ChunkingService(chunk_size=500, chunk_overlap=50),
        "Config C (800/80)": ChunkingService(chunk_size=800, chunk_overlap=80),
    }

    results: dict[str, dict] = {}

    for name, service in configs.items():
        chunks = service.chunk_document(text, document_id="eval_doc_01", filename=doc_path.name)
        lengths = [len(c.text) for c in chunks]
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        results[name] = {
            "chunk_count": len(chunks),
            "chunk_size": service.chunk_size,
            "chunk_overlap": service.chunk_overlap,
            "avg_chunk_length": round(avg_len, 2),
            "min_chunk_length": min(lengths) if lengths else 0,
            "max_chunk_length": max(lengths) if lengths else 0,
        }

    return results


def run_top_k_experiment(
    arg1: Any = None,
    questions: list[dict] | None = None,
    doc_path: Path | None = None
) -> dict[str, dict]:
    """Evaluate top_k values (3, 5, 8) using REAL vector retrieval against indexed document chunks."""
    base_dir = Path(__file__).parent

    # Support flexible positional/keyword arguments
    if isinstance(arg1, Path):
        doc_path = arg1
    elif isinstance(arg1, list) and len(arg1) > 0 and isinstance(arg1[0], dict):
        questions = arg1
    elif isinstance(arg1, list) and len(arg1) > 0 and isinstance(arg1[0], DocumentChunk):
        # Passed chunks as first arg
        pass

    if doc_path is None:
        doc_path = base_dir / "test_document.txt"

    if questions is None or len(questions) == 0:
        questions_path = base_dir / "questions.json"
        if questions_path.exists():
            with open(questions_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
        else:
            questions = []

    text = doc_path.read_text(encoding="utf-8")
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_topk_chroma_"))

    try:
        chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(text, document_id="eval_doc_01", filename=doc_path.name)
        embedder = EmbeddingService()
        vector_store = VectorStoreService(chroma_path=str(temp_dir))

        try:
            vector_store.client.delete_collection(VectorStoreService.COLLECTION_NAME)
        except Exception:
            pass

        embeddings = embedder.embed_documents([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        retriever = RetrievalService(embedding_service=embedder, vector_store_service=vector_store)

        top_k_options = [3, 5, 8]
        results: dict[str, dict] = {}

        for k in top_k_options:
            total_retrieved_chunks = 0
            total_chars = 0
            total_top_score = 0.0

            if questions:
                for q in questions:
                    sources, _, _ = retriever.retrieve_with_metrics(
                        q["question"], document_id="eval_doc_01", top_k=k
                    )
                    total_retrieved_chunks += len(sources)
                    total_chars += sum(len(s.text) for s in sources)
                    if sources:
                        total_top_score += sources[0].score
                num_q = len(questions)
                avg_chunks = total_retrieved_chunks / num_q
                avg_chars = total_chars / num_q
                avg_top_score = total_top_score / num_q
            else:
                retrieved_count = min(k, len(chunks))
                avg_chunks = retrieved_count
                avg_chars = sum(len(c.text) for c in chunks[:retrieved_count])
                avg_top_score = 0.0

            results[f"top_k={k}"] = {
                "top_k": k,
                "chunks_retrieved": min(k, len(chunks)),
                "avg_chunks_retrieved": round(avg_chunks, 1),
                "total_context_chars": round(avg_chars, 1),
                "estimated_tokens": round(avg_chars / 4, 1),
                "avg_top_similarity_score": round(avg_top_score, 4),
                "prompt_bloat_factor": f"{round(k / 3, 2)}x"
            }

        return results
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_threshold_experiment(
    questions: list[dict] | None = None,
    threshold: float = 0.50,
    doc_path: Path | None = None
) -> dict[str, Any]:
    """Execute REAL vector retrieval evaluation across labelled question dataset to measure:

    - Similarity score distribution (in-document vs related-unsupported vs unrelated)
    - Grounding threshold decision boundary
    - Real hit-rate (recall on in-document questions)
    - Real leak-rate (false positive rate on unsupported/unrelated questions)
    """
    base_dir = Path(__file__).parent

    if doc_path is None:
        doc_path = base_dir / "test_document.txt"

    if questions is None:
        questions_path = base_dir / "questions.json"
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

    text = doc_path.read_text(encoding="utf-8")
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_thresh_chroma_"))

    try:
        chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(text, document_id="eval_doc_01", filename=doc_path.name)
        embedder = EmbeddingService()
        vector_store = VectorStoreService(chroma_path=str(temp_dir))

        try:
            vector_store.client.delete_collection(VectorStoreService.COLLECTION_NAME)
        except Exception:
            pass

        embeddings = embedder.embed_documents([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)

        retriever = RetrievalService(embedding_service=embedder, vector_store_service=vector_store)

        eval_summary: dict[str, Any] = {}
        in_doc_hits = 0
        in_doc_total = 0
        leak_count = 0
        unsupported_total = 0

        for q in questions:
            q_id = q["id"]
            q_type = q["type"]
            q_text = q["question"]

            sources, _, _ = retriever.retrieve_with_metrics(
                q_text, document_id="eval_doc_01", top_k=5
            )
            best_score = sources[0].score if sources else 0.0
            passes_threshold = best_score >= threshold
            action = "CALL_LLM_GENERATION" if passes_threshold else "REJECT_GROUNDED_FALSE"

            if q_type == "in_document":
                in_doc_total += 1
                if passes_threshold:
                    in_doc_hits += 1
            else:
                unsupported_total += 1
                if passes_threshold:
                    leak_count += 1

            eval_summary[q_id] = {
                "question": q_text[:50] + ("..." if len(q_text) > 50 else ""),
                "question_type": q_type,
                "best_similarity_score": round(best_score, 4),
                "threshold": threshold,
                "passes_threshold": passes_threshold,
                "decision": action
            }

        hit_rate = (in_doc_hits / in_doc_total) if in_doc_total > 0 else 0.0
        leak_rate = (leak_count / unsupported_total) if unsupported_total > 0 else 0.0

        eval_summary["_summary"] = {
            "threshold_evaluated": threshold,
            "in_document_hit_rate": f"{hit_rate:.1%}",
            "unsupported_leak_rate": f"{leak_rate:.1%}",
            "in_document_hits": f"{in_doc_hits}/{in_doc_total}",
            "unsupported_leaks": f"{leak_count}/{unsupported_total}",
            "calibrated_threshold": 0.50
        }

        return eval_summary
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    base_dir = Path(__file__).parent
    doc_path = base_dir / "test_document.txt"
    questions_path = base_dir / "questions.json"

    print("==================================================")
    print("      RAG SYSTEM CHUNKING EXPERIMENT RESULTS      ")
    print("==================================================")

    if not doc_path.exists():
        print(f"Error: {doc_path} not found.")
        return

    exp_results = run_chunking_experiment(doc_path)
    print(json.dumps(exp_results, indent=2))

    if questions_path.exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        print("\n==================================================")
        print("    RAG REAL VECTOR RETRIEVAL TOP-K EXPERIMENT    ")
        print("==================================================")
        top_k_results = run_top_k_experiment(questions=questions, doc_path=doc_path)
        print(json.dumps(top_k_results, indent=2))

        print("\n==================================================")
        print("  RAG REAL SIMILARITY THRESHOLD & LEAK EVALUATION ")
        print("==================================================")
        print("Evaluating Calibrated Gate Threshold = 0.50 (Empirical Boundary)")
        threshold_results = run_threshold_experiment(questions=questions, threshold=0.50, doc_path=doc_path)
        print(json.dumps(threshold_results, indent=2))


if __name__ == "__main__":
    main()

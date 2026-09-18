import json
from pathlib import Path
from app.services.chunking import ChunkingService, DocumentChunk


def run_chunking_experiment(doc_path: Path) -> dict[str, dict]:
    """Execute chunking comparison across 3 configurations:

    Config A: size=300, overlap=30
    Config B: size=500, overlap=50
    Config C: size=800, overlap=80
    """
    text = doc_path.read_text(encoding="utf-8")

    configs = {
        "Config A (300/30)": ChunkingService(chunk_size=300, chunk_overlap=30),
        "Config B (500/50)": ChunkingService(chunk_size=500, chunk_overlap=50),
        "Config C (800/80)": ChunkingService(chunk_size=800, chunk_overlap=80),
    }

    results: dict[str, dict] = {}

    for name, service in configs.items():
        chunks = service.chunk_document(text, document_id="eval_doc_01", filename="test_document.txt")
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


def run_top_k_experiment(chunks: list[DocumentChunk], questions: list[dict]) -> dict[str, dict]:
    """Evaluate top_k values (3, 5, 8) measuring context character length and chunk count."""
    top_k_options = [3, 5, 8]
    results: dict[str, dict] = {}

    for k in top_k_options:
        retrieved_count = min(k, len(chunks))
        top_chunks = chunks[:retrieved_count]
        total_chars = sum(len(c.text) for c in top_chunks)
        avg_chars_per_prompt = round(total_chars, 2)

        results[f"top_k={k}"] = {
            "top_k": k,
            "chunks_retrieved": retrieved_count,
            "total_context_chars": total_chars,
            "estimated_tokens": round(total_chars / 4, 1),
            "prompt_bloat_factor": f"{round(retrieved_count / 3, 2)}x"
        }

    return results


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

        chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(
            doc_path.read_text(encoding="utf-8"),
            document_id="eval_doc_01",
            filename="test_document.txt"
        )

        print("\n==================================================")
        print("      RAG SYSTEM TOP-K EXPERIMENT RESULTS        ")
        print("==================================================")
        top_k_results = run_top_k_experiment(chunks, questions)
        print(json.dumps(top_k_results, indent=2))


if __name__ == "__main__":
    main()

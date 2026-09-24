import json
from pathlib import Path
from evaluation.evaluate import run_threshold_experiment


def test_threshold_experiment_decision_boundary() -> None:
    """Verify calibrated 0.50 similarity threshold decision boundary filters out unsupported and unrelated questions without leaking."""
    questions_path = Path(__file__).parent.parent / "evaluation" / "questions.json"
    doc_path = Path(__file__).parent.parent / "evaluation" / "test_document.txt"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = run_threshold_experiment(questions, threshold=0.50, doc_path=doc_path)

    assert "q1" in results
    assert "q5" in results
    assert "q7" in results

    # In-document questions (q1..q4) must pass threshold 0.50
    for q_id in ["q1", "q2", "q3", "q4"]:
        assert results[q_id]["passes_threshold"] is True
        assert results[q_id]["decision"] == "CALL_LLM_GENERATION"
        assert results[q_id]["best_similarity_score"] >= 0.50

    # Related unsupported questions (q5..q6) must fail threshold 0.50
    for q_id in ["q5", "q6"]:
        assert results[q_id]["passes_threshold"] is False
        assert results[q_id]["decision"] == "REJECT_GROUNDED_FALSE"
        assert results[q_id]["best_similarity_score"] < 0.50

    # Unrelated questions (q7..q8) must fail threshold 0.50
    for q_id in ["q7", "q8"]:
        assert results[q_id]["passes_threshold"] is False
        assert results[q_id]["decision"] == "REJECT_GROUNDED_FALSE"
        assert results[q_id]["best_similarity_score"] < 0.50

    # Check metrics summary
    assert results["_summary"]["in_document_hit_rate"] == "100.0%"
    assert results["_summary"]["unsupported_leak_rate"] == "0.0%"

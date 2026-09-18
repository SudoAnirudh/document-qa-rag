from pathlib import Path
from evaluation.evaluate import run_chunking_experiment


def test_chunking_experiment_execution() -> None:
    """Verify that run_chunking_experiment evaluates Configs A, B, and C cleanly."""
    doc_path = Path(__file__).parent.parent / "evaluation" / "test_document.txt"
    assert doc_path.exists()

    results = run_chunking_experiment(doc_path)

    assert "Config A (300/30)" in results
    assert "Config B (500/50)" in results
    assert "Config C (800/80)" in results

    # Config A (300) should yield more chunks than Config B (500) and Config C (800)
    count_a = results["Config A (300/30)"]["chunk_count"]
    count_b = results["Config B (500/50)"]["chunk_count"]
    count_c = results["Config C (800/80)"]["chunk_count"]

    assert count_a >= count_b >= count_c
    assert count_c > 0

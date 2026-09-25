from pathlib import Path


def test_model_artifacts_exist_after_training() -> None:
    model_path = Path("models/spam_classifier.joblib")
    metrics_path = Path("models/spam_classifier.metrics.json")
    assert model_path.exists(), "Model artifact not found. Run training first."
    assert metrics_path.exists(), "Metrics artifact not found. Run training first."


def test_analysis_pipeline_handles_missing_topic_models() -> None:
    from src.engine import analyze_email_pipeline

    result = analyze_email_pipeline("Thanks for the quick support and help on this issue.")

    assert isinstance(result, dict)
    assert "classification" in result
    assert "key_phrases" in result

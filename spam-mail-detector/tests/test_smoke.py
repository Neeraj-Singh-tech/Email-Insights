from pathlib import Path


def test_model_artifacts_exist_after_training() -> None:
    model_path = Path("models/spam_classifier.joblib")
    metrics_path = Path("models/spam_classifier.metrics.json")
    assert model_path.exists(), "Model artifact not found. Run training first."
    assert metrics_path.exists(), "Metrics artifact not found. Run training first."

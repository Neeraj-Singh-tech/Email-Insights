from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_sessionstart(session) -> None:
    model_path = Path("models/spam_classifier.joblib")
    metrics_path = Path("models/spam_classifier.metrics.json")

    if model_path.exists() and metrics_path.exists():
        return

    subprocess.run(
        [
            sys.executable,
            "src/train.py",
            "--data",
            "data/spam_ham_dataset.csv",
            "--model-out",
            str(model_path),
        ],
        check=True,
    )

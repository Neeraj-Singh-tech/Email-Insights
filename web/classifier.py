from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class SpamClassifier:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Trained model was not found at {self.model_path}. Train the model first."
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, subject: str, body: str) -> dict[str, float | str]:
        model = self._load()
        text = f"Subject: {subject}\n{body}".strip()

        pred = int(model.predict([text])[0])
        probs = model.predict_proba([text])[0]

        label = "spam" if pred == 1 else "ham"
        return {
            "label": label,
            "ham_probability": float(probs[0]),
            "spam_probability": float(probs[1]),
        }

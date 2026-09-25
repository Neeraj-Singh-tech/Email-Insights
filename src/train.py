import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop auto-generated unnamed index columns if present.
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in dataset.")

    if "label" in df.columns:
        df["target"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    elif "label_num" in df.columns:
        df["target"] = pd.to_numeric(df["label_num"], errors="coerce")
    else:
        raise ValueError("Expected either 'label' or 'label_num' column in dataset.")

    df["text"] = df["text"].fillna("").astype(str)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    return df[["text", "target"]]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spam/ham email classifier.")
    parser.add_argument(
        "--data",
        default="data/spam_ham_dataset.csv",
        help="Path to CSV dataset containing text and label columns.",
    )
    parser.add_argument(
        "--model-out",
        default="models/spam_classifier.joblib",
        help="Where to save the trained model pipeline.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["target"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    joblib.dump(pipeline, model_out)

    metrics_out = model_out.with_suffix(".metrics.json")
    metrics_payload = {
        "samples_total": int(len(df)),
        "samples_train": int(len(X_train)),
        "samples_test": int(len(X_test)),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Model saved to: {model_out}")
    print(f"Metrics saved to: {metrics_out}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import joblib


def read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Provide either --text or --file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict if an email is spam or ham.")
    parser.add_argument(
        "--model",
        default="models/spam_classifier.joblib",
        help="Path to trained joblib model.",
    )
    parser.add_argument("--text", help="Raw email text.")
    parser.add_argument("--file", help="Path to a text file containing email content.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    text = read_text(args)
    model = joblib.load(model_path)

    pred = int(model.predict([text])[0])
    probs = model.predict_proba([text])[0]

    label = "spam" if pred == 1 else "ham"
    print(f"Prediction: {label}")
    print(f"Probabilities -> ham: {probs[0]:.4f}, spam: {probs[1]:.4f}")


if __name__ == "__main__":
    main()

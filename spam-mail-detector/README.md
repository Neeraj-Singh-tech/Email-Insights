# Spam Mail Detector (ML + Web Inbox)

This project trains a spam-vs-ham classifier from your mail dataset using a classic text ML pipeline:
- TF-IDF vectorization (word unigrams + bigrams)
- Logistic Regression classifier

## Project Structure

- `data/spam_ham_dataset.csv` - dataset copy
- `src/train.py` - training + evaluation + model export
- `src/predict.py` - single-email prediction CLI
- `web/app.py` - FastAPI backend for email routing API
- `web/templates/index.html` - inbox-style frontend
- `web/static/` - styles and JavaScript for UI
- `models/` - trained model artifacts

## 1) Setup

Install Python 3.10+ and then run:

```powershell
cd C:\Users\Admin\projects\spam-mail-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Train the model

```powershell
python src\train.py --data data\spam_ham_dataset.csv --model-out models\spam_classifier.joblib
```

Outputs:
- `models/spam_classifier.joblib`
- `models/spam_classifier.metrics.json`

## 3) Predict new emails

Using direct text:

```powershell
python src\predict.py --model models\spam_classifier.joblib --text "Win a free prize now! Click here."
```

Using a file:

```powershell
python src\predict.py --model models\spam_classifier.joblib --file sample_mail.txt
```

## 4) Run Full-Stack Inbox Demo (Option 2)

Start the backend + frontend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```

Or use:

```powershell
scripts\run_web.ps1
```

Then open:
- `http://127.0.0.1:8000`

How it works:
- You add a mail from the form.
- Backend runs your trained spam model.
- If spam score is high, it appears in Spam folder.
- Otherwise it appears in Primary inbox.
- Use "Load Demo Mails" to quickly populate a realistic inbox.

## Notes

- If classes are imbalanced, classifier uses `class_weight=balanced`.
- You can later improve performance with model comparison (Linear SVM, Naive Bayes), hyperparameter search, and threshold tuning.

Set-Location "$PSScriptRoot\.."
.\.venv\Scripts\python.exe src\train.py --data data\spam_ham_dataset.csv --model-out models\spam_classifier.joblib

Set-Location "$PSScriptRoot\.."
.\.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload

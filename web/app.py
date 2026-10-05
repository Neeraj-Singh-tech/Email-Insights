from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.engine import analyze_email_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmailSubmission(BaseModel):
    sender: str = Field(min_length=3, max_length=100)
    subject: str = Field(min_length=1, max_length=180)
    body: str = Field(min_length=1, max_length=5000)


class EmailAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


app = FastAPI(title="Email-Insights Dashboard")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))

INBOX: list[dict] = []


def enrich_email(subject: str, sender: str, body: str) -> dict:
    analysis = analyze_email_pipeline(f"{subject} {body}")
    classification = analysis.get("classification", {})
    return {
        "id": len(INBOX) + 1,
        "subject": subject,
        "sender": sender,
        "body": body,
        "is_spam": analysis.get("is_spam", False),
        "topic": classification.get("topic", "General"),
        "priority": classification.get("priority", "P3 - Normal"),
        "overall_tone": classification.get("overall_tone", "Neutral"),
        "action_items": analysis.get("action_items", []),
        "key_phrases": analysis.get("key_phrases", []),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"request": request, "emails": INBOX})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze_email(request: EmailAnalysisRequest) -> dict:
    from src.engine import analyze_email_pipeline

    return analyze_email_pipeline(request.text)


@app.get("/api/emails")
def list_emails() -> list[dict]:
    return INBOX


@app.post("/api/emails")
def add_email(email: EmailSubmission) -> dict:
    enriched = enrich_email(email.subject, email.sender, email.body)
    INBOX.insert(0, enriched)
    return {"message": "Email processed", "email": enriched}


@app.post("/api/seed")
def seed_demo_data() -> dict[str, str | int]:
    demo_data = [
        {"sender": "alerts@bank-notify.com", "subject": "Urgent: verify your account now", "body": "Suspicious login attempt. Click this link and verify your account within 10 minutes."},
        {"sender": "manager@company.com", "subject": "Team standup moved to 10:30", "body": "Please join the conference room by 10:30. Bring your sprint updates."},
        {"sender": "promo@deals-center.net", "subject": "You won a free laptop", "body": "Claim your reward now. Limited seats. Share bank details to get delivery."},
        {"sender": "hr@company.com", "subject": "Leave policy update", "body": "The updated leave policy document is attached. Please review before Friday."},
    ]

    INBOX.clear()
    for item in demo_data:
        INBOX.append(enrich_email(item["subject"], item["sender"], item["body"]))

    return {"status": "seeded", "count": len(INBOX)}

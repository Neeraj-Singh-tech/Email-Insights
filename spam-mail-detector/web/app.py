from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from web.classifier import SpamClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"

classifier = SpamClassifier(MODEL_PATH)


class IncomingEmail(BaseModel):
    sender: str = Field(min_length=3, max_length=100)
    subject: str = Field(min_length=1, max_length=180)
    body: str = Field(min_length=1, max_length=5000)


class StoredEmail(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    received_at: str
    bucket: Literal["primary", "spam"]
    spam_probability: float
    ham_probability: float


app = FastAPI(title="Spam Mail Detector Inbox", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "web" / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "web" / "templates"))

INBOX: list[StoredEmail] = []


def route_email(payload: IncomingEmail) -> StoredEmail:
    result = classifier.predict(payload.subject, payload.body)
    bucket: Literal["primary", "spam"] = "spam" if result["label"] == "spam" else "primary"

    return StoredEmail(
        id=str(uuid4()),
        sender=payload.sender,
        subject=payload.subject,
        body=payload.body,
        received_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        bucket=bucket,
        spam_probability=float(result["spam_probability"]),
        ham_probability=float(result["ham_probability"]),
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/emails")
def list_emails() -> dict[str, list[dict]]:
    primary = [mail.model_dump() for mail in INBOX if mail.bucket == "primary"]
    spam = [mail.model_dump() for mail in INBOX if mail.bucket == "spam"]
    return {"primary": primary, "spam": spam}


@app.post("/api/emails")
def add_email(email: IncomingEmail) -> dict:
    try:
        routed = route_email(email)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    INBOX.insert(0, routed)
    return {"message": "Email processed", "email": routed.model_dump()}


@app.post("/api/seed")
def seed_demo_data() -> dict[str, int]:
    demo_data = [
        IncomingEmail(
            sender="alerts@bank-notify.com",
            subject="Urgent: verify your account now",
            body="Suspicious login attempt. Click this link and verify your account within 10 minutes.",
        ),
        IncomingEmail(
            sender="manager@company.com",
            subject="Team standup moved to 10:30",
            body="Please join the conference room by 10:30. Bring your sprint updates.",
        ),
        IncomingEmail(
            sender="promo@deals-center.net",
            subject="You won a free laptop",
            body="Claim your reward now. Limited seats. Share bank details to get delivery.",
        ),
        IncomingEmail(
            sender="hr@company.com",
            subject="Leave policy update",
            body="The updated leave policy document is attached. Please review before Friday.",
        ),
    ]

    added = 0
    for item in demo_data:
        INBOX.insert(0, route_email(item))
        added += 1

    return {"added": added}

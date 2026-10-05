import os
from pathlib import Path

import joblib
import numpy as np
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Initialize spaCy in a way that still works if the model was not downloaded.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

# 2. Load RoBERTa directly from Hugging Face
HF_REPO = "dextube/email-emotion-roberta"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading emotion model from Hugging Face ({HF_REPO}) on {device}...")
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
emotion_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
emotion_model.to(device)
emotion_model.eval()

# 3. Load Local Classifiers
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

topic_vectorizer = None
topic_clf = None
vectorizer_path = MODELS_DIR / "topic_vectorizer.joblib"
model_path = MODELS_DIR / "topic_model.joblib"
if vectorizer_path.exists() and model_path.exists():
    topic_vectorizer = joblib.load(vectorizer_path)
    topic_clf = joblib.load(model_path)

# Assumes your spam model was trained with scikit-learn
spam_model = joblib.load(MODELS_DIR / "spam_classifier.joblib")

NEGATIVE_EMOTION_LABELS = {
    "anger",
    "annoyance",
    "fear",
    "disgust",
    "sadness",
    "anxiety_urgency",
    "frustration_complaint",
}
POSITIVE_EMOTION_LABELS = {"joy", "optimism", "gratitude", "positive_appreciation", "empathy_support"}
ESCALATION_KEYWORDS = (
    "urgent",
    "immediately",
    "asap",
    "critical",
    "emergency",
    "failed",
    "fail",
    "deadline",
    "escalat",
    "before we lose",
    "drop everything",
    "at risk",
)


def _peak_emotion_scores(email_text: str) -> tuple[float, float, bool]:
    doc = nlp(email_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 5]
    if not sentences:
        return 0.0, 0.0, False

    negative_peak = 0.0
    positive_peak = 0.0
    lower_text = email_text.lower()
    explicit_escalation = any(keyword in lower_text for keyword in ESCALATION_KEYWORDS)

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        if hasattr(emotion_model, "config") and hasattr(emotion_model.config, "id2label"):
            score_map = {
                str(emotion_model.config.id2label[idx]).lower(): float(prob)
                for idx, prob in enumerate(probabilities)
            }
        else:
            score_map = {label.lower(): float(prob) for label, prob in zip(["neutral"], probabilities[:1])}

        negative_sentence_score = max(
            (score_map.get(label, 0.0) for label in NEGATIVE_EMOTION_LABELS),
            default=0.0,
        )
        positive_sentence_score = max(
            (score_map.get(label, 0.0) for label in POSITIVE_EMOTION_LABELS),
            default=0.0,
        )

        negative_peak = max(negative_peak, negative_sentence_score)
        positive_peak = max(positive_peak, positive_sentence_score)

    return negative_peak, positive_peak, explicit_escalation


def _fallback_topic(email_text: str) -> str:
    lowered = email_text.lower()
    if any(keyword in lowered for keyword in ["invoice", "payment", "refund", "billing", "charge"]):
        return "billing"
    if any(keyword in lowered for keyword in ["team", "project", "meeting", "deadline", "schedule"]):
        return "work"
    if any(keyword in lowered for keyword in ["offer", "winner", "click", "free", "prize", "claim"]):
        return "marketing"
    if any(keyword in lowered for keyword in ["thanks", "appreciate", "happy", "support", "help"]):
        return "support"
    return "general"


def analyze_email_pipeline(email_text: str):
    # Phase 0: Spam Detection
    is_spam = bool(spam_model.predict([email_text])[0] == 1)
    if is_spam:
        return {"is_spam": True, "status": "filtered", "message": "Flagged as SPAM by baseline filter"}

    doc = nlp(email_text)
    peak_negative_score, peak_positive_score, explicit_escalation = _peak_emotion_scores(email_text)

    if peak_negative_score >= 0.60 or explicit_escalation:
        tone = "Negative / Escalated"
        priority = "P1 - Urgent"
    elif 0.35 <= peak_negative_score < 0.60:
        tone = "Negative / Escalated"
        priority = "P2 - High"
    elif peak_positive_score >= 0.50 and peak_negative_score < 0.35:
        tone = "Positive"
        priority = "P3 - Normal"
    else:
        tone = "Neutral"
        priority = "P3 - Normal"

    if topic_vectorizer is not None and topic_clf is not None:
        vec = topic_vectorizer.transform([email_text])
        topic = topic_clf.predict(vec)[0]
        topic_conf = float(np.max(topic_clf.predict_proba(vec)[0]))
    else:
        topic = _fallback_topic(email_text)
        topic_conf = 0.55

    entities = [
        {"text": ent.text, "type": ent.label_}
        for ent in doc.ents
        if ent.label_ in {"MONEY", "DATE", "TIME", "PERSON", "ORG"}
    ]

    phrases = list(
        set(
            [
                chunk.text.strip().lower()
                for chunk in doc.noun_chunks
                if len(chunk.text.split()) <= 3 and chunk.text.lower() not in ["it", "we", "you", "they", "i"]
            ]
        )
    )[:5]

    return {
        "is_spam": False,
        "classification": {
            "topic": topic,
            "topic_confidence": round(topic_conf, 4),
            "priority": priority,
            "overall_tone": tone,
        },
        "action_items": entities,
        "key_phrases": phrases,
    }
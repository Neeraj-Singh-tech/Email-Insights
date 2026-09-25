import os
import joblib
import spacy
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# 2. Load RoBERTa directly from Hugging Face
HF_REPO = "dextube/email-emotion-roberta"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading emotion model from Hugging Face ({HF_REPO}) on {device}...")
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
emotion_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
emotion_model.to(device)
emotion_model.eval()

# 3. Load Local Classifiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

topic_vectorizer = joblib.load(os.path.join(MODELS_DIR, "topic_vectorizer.joblib"))
topic_clf = joblib.load(os.path.join(MODELS_DIR, "topic_model.joblib"))

# Assumes your spam model was trained with scikit-learn
spam_model = joblib.load(os.path.join(MODELS_DIR, "spam_classifier.joblib"))

EMOTION_LABELS = [
    'positive_appreciation', 'frustration_complaint', 'anxiety_urgency',
    'inquiry_curiosity', 'empathy_support', 'surprise_realization', 'neutral'
]

def analyze_email_pipeline(email_text: str):
    # Phase 0: Spam Detection
    # Adjust the predict call if your spam model outputs something other than [1] for spam
    is_spam = bool(spam_model.predict([email_text])[0] == 1)
    if is_spam:
        return {
            "is_spam": True,
            "status": "filtered",
            "message": "Flagged as SPAM by baseline filter"
        }

    # Phases 1 & 2: Emotion & Urgency (Smart Reader Peak Pooling)
    doc = nlp(email_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
    if not sentences:
        sentences = [email_text]

    max_probs = {label: 0.0 for label in EMOTION_LABELS}
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(emotion_model(**inputs).logits).cpu().numpy()[0]
        
        for i, label in enumerate(EMOTION_LABELS):
            if probs[i] > max_probs[label]:
                max_probs[label] = float(probs[i])

    detected = {}
    if max_probs['anxiety_urgency'] >= 0.28:
        detected['anxiety_urgency'] = max_probs['anxiety_urgency']
    if max_probs['frustration_complaint'] >= 0.28:
        detected['frustration_complaint'] = max_probs['frustration_complaint']
    if max_probs['positive_appreciation'] >= 0.40:
        detected['positive_appreciation'] = max_probs['positive_appreciation']

    has_neg = 'frustration_complaint' in detected or 'anxiety_urgency' in detected
    has_pos = 'positive_appreciation' in detected

    if has_neg:
        tone = "Negative / Escalated"
        priority = "P1 - Urgent" if 'anxiety_urgency' in detected else "P2 - High"
    elif has_pos:
        tone = "Positive / Appreciative"
        priority = "P3 - Normal"
    else:
        tone = "Neutral / Standard"
        priority = "P3 - Normal"

    # Phase 3: Topic Classification (TF-IDF)
    vec = topic_vectorizer.transform([email_text])
    topic = topic_clf.predict(vec)[0]
    topic_conf = float(np.max(topic_clf.predict_proba(vec)[0]))

    # Phase 4: Action Items & Keywords (spaCy NER)
    entities = [
        {"text": ent.text, "type": ent.label_}
        for ent in doc.ents
        if ent.label_ in {"MONEY", "DATE", "TIME", "PERSON", "ORG"}
    ]
    
    phrases = list(set([
        chunk.text.strip().lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) <= 3 and chunk.text.lower() not in ["it", "we", "you", "they", "i"]
    ]))[:5]

    return {
        "is_spam": False,
        "classification": {
            "topic": topic,
            "topic_confidence": round(topic_conf, 4),
            "priority": priority,
            "overall_tone": tone
        },
        "action_items": entities,
        "key_phrases": phrases
    }
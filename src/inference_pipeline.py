import random
import joblib
from transformers import pipeline

from src.text_preprocessing import clean_text

urgency_model = joblib.load("models/urgency_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

URGENCY_SCORE = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

SENTIMENT_SCORE = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}

def analyze_ticket(ticket_text: str) -> dict:
    cleaned = clean_text(ticket_text)
    vec = vectorizer.transform([cleaned])

    urgency = urgency_model.predict(vec)[0]
    sentiment_raw = sentiment_pipeline(ticket_text[:512])[0]
    sentiment = "Positive" if sentiment_raw["label"] == "POSITIVE" else "Negative"
    age_hours = random.randint(1, 72)
    priority_score = (
        URGENCY_SCORE.get(urgency, 1) * 3
        + SENTIMENT_SCORE.get(sentiment, 1) * 2
        + min(age_hours / 24, 2)
    )

    return {
        "urgency": urgency,
        "sentiment": sentiment,
        "priority_score": round(priority_score, 2)
    }

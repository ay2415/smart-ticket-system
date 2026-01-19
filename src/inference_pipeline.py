import torch
import random

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from src.text_preprocessing import clean_text



URGENCY_MODEL_PATH = "models/urgency_transformer"

urgency_tokenizer = DistilBertTokenizerFast.from_pretrained(
    URGENCY_MODEL_PATH
)

urgency_model = DistilBertForSequenceClassification.from_pretrained(
    URGENCY_MODEL_PATH
)

urgency_model.eval()



def predict_urgency(text: str) -> str:
    inputs = urgency_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = urgency_model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()

    id_to_label = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return id_to_label[prediction_id]




def predict_sentiment(text: str) -> str:
    negative_words = ["fail", "error", "not working", "urgent", "crash", "issue"]

    text_lower = text.lower()
    for w in negative_words:
        if w in text_lower:
            return "Negative"

    return "Positive"




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


def calculate_priority(urgency, sentiment, age_hours):
    return (
        URGENCY_SCORE.get(urgency, 1) * 3
        + SENTIMENT_SCORE.get(sentiment, 1) * 2
        + min(age_hours / 24, 2)
    )




def analyze_ticket(text: str):
    cleaned_text = clean_text(text)
    urgency = predict_urgency(cleaned_text)
    sentiment = predict_sentiment(cleaned_text)
    age_hours = random.randint(1, 72)
    priority_score = round(
        calculate_priority(urgency, sentiment, age_hours), 2
    )

    return {
        "urgency": urgency,
        "sentiment": sentiment,
        "priority_score": priority_score
    }

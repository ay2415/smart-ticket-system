import pandas as pd
from transformers import pipeline

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

df = pd.read_csv("../data/tickets_cleaned.csv", encoding="latin1")

def get_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]
    if result["label"] == "POSITIVE":
        return "Positive"
    else:
        return "Negative"

df["sentiment"] = df["clean_text"].apply(get_sentiment)

df.to_csv("../data/tickets_with_sentiment.csv", index=False)

print(df[["clean_text", "sentiment"]].head())

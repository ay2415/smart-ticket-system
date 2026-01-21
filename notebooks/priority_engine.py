import pandas as pd
import random


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


def cal_priority(urgency,sentiment,age_hours):
    return(
        URGENCY_SCORE.get(urgency,1)*3+SENTIMENT_SCORE.get(sentiment,1)*2 + min(age_hours/24,2)
        )

df = pd.read_csv("../data/tickets_with_sentiment.csv", encoding="latin1")

df["age_hours"] = [random.randint(1, 72) for _ in range(len(df))]

df["priority_score"] = df.apply(
    lambda row: cal_priority(
        row["urgency"],
        row["sentiment"],
        row["age_hours"]
    ),
    axis=1
)

df_sorted = df.sort_values("priority_score", ascending=False)

print(df_sorted[["clean_text","urgency","sentiment","priority_score"]].head())

import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

df = pd.read_csv("data/urgency_train_en.csv")

label_map = {"low": 0, "medium": 1, "high": 2}

df["label_id"] = df["label"].map(label_map)

test_df = df.sample(frac=0.1, random_state=42)

MODEL_PATH = "models/urgency_transformer"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

y_true = []
y_pred = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    inputs = tokenizer(
        row["text"],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    y_true.append(row["label_id"])
    y_pred.append(pred)

print(classification_report(y_true, y_pred, target_names=["Low", "Medium", "High"]))
print(confusion_matrix(y_true, y_pred))

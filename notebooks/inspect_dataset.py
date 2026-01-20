import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch


df = pd.read_csv("../data/urgency_train_en.csv")

print("Dataset size:", len(df))
print(df["label"].value_counts())

label_map = {"low": 0, "medium": 1, "high": 2}
df["label"] = df["label"].map(label_map)


train_df = df.sample(frac=0.9, random_state=42)
val_df = df.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)


tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="../models/urgency_transformer",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=200,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()


trainer.save_model("../models/urgency_transformer")
tokenizer.save_pretrained("../models/urgency_transformer")

print("Transformer urgency model trained and saved")

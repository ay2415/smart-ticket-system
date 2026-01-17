import spacy
import pandas as pd
import re
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    doc= nlp(text)
    cleaned_words = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha]
    return " ".join(cleaned_words)
if __name__ == "__main__":
    input_path = "../data/tickets.csv"
    output_path = "../data/tickets_cleaned.csv"
    df = pd.read_csv(input_path, encoding="latin1")

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column")

    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(" Cleaned dataset saved to:", output_path)

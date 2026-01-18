import spacy
import re
import pandas as pd

nlp=spacy.load("en_core_web_sm")

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text=text.lower()

    text=re.sub(r"[^a-zA-Z\s]","",text)

    doc=nlp(text)

    cleaned_words=[
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha
    ]

    return " ".join(cleaned_words)
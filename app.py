import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
from src.inference_pipeline import analyze_ticket

st.title("Smart Ticket Prioritization")

text = st.text_area("Enter support ticket text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        result = analyze_ticket(text)
        st.write("Urgency:", result["urgency"])
        st.write("Sentiment:", result["sentiment"])
        st.write("Priority Score:", round(result["priority_score"], 2))

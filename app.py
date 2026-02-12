import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
from src.inference_pipeline import analyze_ticket

st.set_page_config(
    page_title="Smart Ticket Prioritization",
    layout="centered"
)

st.markdown(
    unsafe_allow_html=True
)
st.divider()
text = st.text_area(
    "Support Ticket",
    placeholder="Describe the issue here...",
    height=160
)
analyze =st.button("Analyze",use_container_width=True)
st.divider()
if analyze:
    if not text.strip():
        st.warning("Please enter ticket text.")
    else:
        with st.spinner("Processing..."):
            result =analyze_ticket(text)
        urgency =result["urgency"]
        sentiment= result["sentiment"]
        score=float(result["priority_score"])

        col1,col2,col3=st.columns(3)

        col1.metric("Urgency",urgency)
        col2.metric("Sentiment",sentiment)
        col3.metric("Priority Score",f"{score:.2f}")

        st.markdown("<br>",unsafe_allow_html=True)  

        normalized_score=min(max(score/15,0),1)

        st.progress(normalized_score)

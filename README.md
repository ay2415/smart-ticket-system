## Smart Support Ticket Prioritization System

This project is an end-to-end machine learning system that analyzes customer support tickets and helps decide which issues should be handled first.

Given a support ticket written in plain text, the system predicts the urgency of the issue, the sentiment of the message, and an overall priority score. The goal is to reduce manual effort in ticket priority and ensure that critical issues are addressed quickly.



##### What the system does

The workflow of the system is straightforward

1. A customer support ticket is provided as text input
2. The text is cleaned and preprocessed
3. Urgency is predicted using a fine-tuned DistilBERT model
4. Sentiment is predicted using a transformer-based sentiment model
5. Urgency, sentiment, and ticket age are combined into a priority score
6. Results are made available through an API and a simple web interface



##### Project structure

The project is organised into the following components.

api
FastAPI backend used for real-time inference.

src
Contains text preprocessing logic and the inference pipeline.

models
Stores trained machine learning models used during inference.

data
Contains datasets used for training and evaluation.

notebooks
Includes scripts for training, evaluation, and experimentation.

app.py
A simple Streamlit-based user interface.

README.md



##### Dataset

The urgency model was trained on a large IT support ticket dataset containing realistic customer issues. Only English tickets were used for training. Each ticket is labelled with an urgency level (Low, Medium, High), making the dataset suitable for supervised learning.



##### Models used



Urgency classification
A fine-tuned DistilBERT model trained to classify tickets into Low, Medium, or High urgency.

Sentiment analysis
A transformer-based sentiment model used to determine whether the message is positive, neutral, or negative.



Priority scoring
A rule-based scoring approach that combines urgency, sentiment, and ticket age into a single numeric priority score.



###### Model performance

The urgency classification model was evaluated on a held-out test set. Overall accuracy is approximately 83 percent. Performance is strongest for high-urgency tickets, with precision and recall around 0.77. This is important in real-world support systems where missing critical issues is costly.



##### Running the project

Start the API server by running:

###### python -m uvicorn api.main:app --reload

Open the API documentation in your browser:

###### http://127.0.0.1:8000/docs

Run the Streamlit user interface:

###### streamlit run app.py

Open the interface in your browser:

http://localhost:8501



Paste a support ticket and click Analyze to view the predictions.

###### Example output

Urgency: High
Sentiment: Negative
Priority score: 14.8


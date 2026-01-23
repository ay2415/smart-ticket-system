# **# Smart Support Ticket Prioritization System**

# 

# **## Overview**

# 

# **The Smart Support Ticket Prioritization System is an end-to-end machine learning application designed to analyze customer support tickets and automatically determine which issues should be handled first.**

# 

# **Given a support ticket written in plain text, the system predicts the urgency of the issue, the sentiment of the message, and an overall priority score. The goal is to reduce manual effort in ticket triaging and ensure that critical customer issues are addressed quickly.**

# 

# **---**

# 

# **## What the System Does**

# 

# **The workflow of the system is straightforward:**

# 

# **1. A customer support ticket is provided as text input**  

# **2. The text is cleaned and preprocessed**  

# **3. Urgency is predicted using a fine-tuned DistilBERT model**  

# **4. Sentiment is predicted using a transformer-based sentiment analysis model**  

# **5. Urgency, sentiment, and ticket age are combined into a single priority score**  

# **6. Results are made available through an API and a simple web interface**  

# 

# **---**

# 

# **## Dataset**

# 

# **The urgency model was trained on a large IT support ticket dataset containing realistic customer issues.**

# 

# **- Only English tickets were used for training**  

# **- Each ticket is labeled with an urgency level: Low, Medium, or High**  

# 

# **This dataset is suitable for supervised learning and reflects real-world customer support scenarios.**

# 

# **---**

# 

# **## Models Used**

# 

# **### Urgency Classification**

# **A fine-tuned DistilBERT model trained to classify support tickets into Low, Medium, or High urgency categories.**

# 

# **### Sentiment Analysis**

# **A transformer-based sentiment model used to determine whether the message is Positive, Neutral, or Negative.**

# 

# **### Priority Scoring**

# **A rule-based scoring approach that combines urgency, sentiment, and ticket age into a single numeric priority score.**

# 

# **---**

# 

# **## Model Performance**

# 

# **The urgency classification model was evaluated on a held-out test set.**

# 

# **- Overall accuracy: approximately 83%**  

# **- Performance is strongest for high-urgency tickets**  

# **- Precision and recall for high-urgency tickets are around 0.77**  

# 

# **This is important in real-world support systems where missing critical issues can be costly.**

# 

# **---**

# 

# **## Running the Project**

# 

# **### Start the API Server**

# 

# **```bash**

# **python -m uvicorn api.main:app --reload**

# 


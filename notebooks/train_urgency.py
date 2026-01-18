import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data_path="../data/tickets_cleaned.csv"
df=pd.read_csv(data_path,encoding="latin1")
print(df[["clean_text","urgency"]].head())

X=df["clean_text"]
y=df["urgency"]

vectorizer=TfidfVectorizer(
    max_features=2000,
    ngram_range=(1,2)
)
X_vec=vectorizer.fit_transform(X)
print("Vectorizer shape:",X_vec.shape)

X_train, X_test, y_train, y_test = train_test_split(X_vec,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model, "../models/urgency_model.pkl")
print("Urgency model saved")

y_pred = model.predict(X_test)
print("Urgency Accuracy:", accuracy_score(y_test, y_pred))
print("\nUrgency Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

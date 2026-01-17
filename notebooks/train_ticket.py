import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data_path="../data/tickets_cleaned.csv"
df =pd.read_csv(data_path,encoding="latin1")
print(df.head())

X=df["clean_text"]
y=df["type"]

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X_vectorized = vectorizer.fit_transform(X)

print("Shape of vectorized data:", X_vectorized.shape)


X_train,X_test,y_train,y_test=train_test_split(
    X_vectorized,y,test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

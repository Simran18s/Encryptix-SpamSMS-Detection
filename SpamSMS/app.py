import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Data preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_pred)

# Train a Logistic Regression classifier
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Train a Support Vector Machine classifier
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)
svm_accuracy = accuracy_score(y_test, svm_pred)

# Streamlit app
st.title("SMS Spam Classifier")

st.write("Naive Bayes Accuracy: ", nb_accuracy)
st.write("Logistic Regression Accuracy: ", lr_accuracy)
st.write("SVM Accuracy: ", svm_accuracy)

# User input
user_input = st.text_area("Enter your SMS message:")

if user_input:
    user_input_tfidf = vectorizer.transform([user_input])
    nb_result = nb_model.predict(user_input_tfidf)[0]
    lr_result = lr_model.predict(user_input_tfidf)[0]
    svm_result = svm_model.predict(user_input_tfidf)[0]
    
    st.write("Naive Bayes Prediction: ", "Spam" if nb_result == 1 else "Not Spam")
    st.write("Logistic Regression Prediction: ", "Spam" if lr_result == 1 else "Not Spam")
    st.write("SVM Prediction: ", "Spam" if svm_result == 1 else "Not Spam")

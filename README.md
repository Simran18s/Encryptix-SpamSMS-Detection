# SpamSMS-Detection

# SMS Spam Classifier

This project is an implementation of a machine learning model to classify SMS messages as spam or legitimate (ham). It uses techniques like TF-IDF for feature extraction and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machine (SVM) for the classification task. A Streamlit app is provided for real-time SMS spam classification.

## Dataset

The dataset used is `spam.csv`, which contains the following columns:
- `v1`: Label (ham or spam)
- `v2`: SMS message

## Project Structure

- `sms_spam_classifier.py`: The main file that contains code for data preprocessing, model training, and Streamlit app.
- `spam.csv`: The dataset file.

## Requirements

The following Python libraries are required:
- pandas
- numpy
- scikit-learn
- streamlit

You can install the required libraries using:
```bash
pip install pandas numpy scikit-learn streamlit

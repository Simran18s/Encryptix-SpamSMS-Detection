# SpamSMS-Detection

This project is an implementation of a machine learning model to classify SMS messages as spam or legitimate (ham). It uses techniques like TF-IDF for feature extraction and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machine (SVM) for the classification task.

### Running The model in terminal

    Streamlit run app.py

## Dataset

The dataset used is `spam.csv`, which contains the following columns:
- `v1`: Label (ham or spam)
- `v2`: SMS message

## Project Structure

- `app.py`: The main file that contains code for data preprocessing, model training, and Streamlit app.
- `spam.csv`: The dataset file.

## Requirements

The following Python libraries are required:
- pandas
- numpy
- scikit-learn
- streamlit

install the required libraries using:
```bash
pip install pandas numpy scikit-learn streamlit


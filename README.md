# 📰 News Category Prediction using Machine Learning

## 📌 Overview
This project focuses on automatically classifying news articles into predefined categories using Natural Language Processing (NLP) and Machine Learning techniques.  
The model learns from textual patterns in news content to predict the most appropriate category for unseen articles.

---

## 🎯 Problem Statement
With the massive volume of news generated daily, manual categorization is inefficient and error-prone.  
This project aims to build a machine learning-based solution that can **accurately and efficiently classify news articles** into relevant categories.

---

## 🚀 Features
- Text preprocessing and cleaning
- Feature extraction using NLP techniques
- Supervised machine learning model for classification
- Category prediction for new/unseen news articles
- End-to-end implementation in a single Jupyter Notebook

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **NLP:** TF-IDF Vectorization  
- **Model:** Traditional ML Classifier (Naive Bayes / Logistic Regression)  
- **Environment:** Jupyter Notebook  

---

## 🧠 How It Works
1. Load the news dataset (`cat_3_news.csv`).
2. Clean and preprocess text data (lowercasing, stopword removal, tokenization).
3. Convert text into numerical features using TF-IDF.
4. Train a supervised classification model.
5. Evaluate model performance using standard metrics.
6. Predict categories for new news articles.

---

## 📂 Project Structure
├── cat_3_news.csv
├── news_category_prediction (1).ipynb
└── README.md

📊 Dataset Description
File: cat_3_news.csv
Contains news text along with their corresponding categories.
Used for training and evaluating the classification model.


📊 Output
The trained model predicts the category of a news article based on its textual content.
Output is displayed directly in the notebook as:
Predicted news category label
Model evaluation metrics such as accuracy, precision, recall, and F1-score.

Sample Output:
Input News: "Government announces new economic policy..."
Predicted Category: Politics
Evaluation results help assess how well the model generalizes to unseen data.

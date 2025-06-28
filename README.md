# 🕵️‍♂️ Fake News Detection System

A machine learning-powered web application that classifies news articles as either **FAKE** or **REAL** using Natural Language Processing techniques and multiple classification algorithms.


## 🎯 Overview

This project implements a complete pipeline for fake news detection, from data preprocessing to model deployment. It uses two different machine learning algorithms (Naive Bayes and Logistic Regression) combined with two text vectorization methods (TF-IDF and Count Vectorizer) to provide accurate classification results.

### Key Components:
- **Data preprocessing and cleaning**
- **Feature extraction using TF-IDF and Count Vectorization**
- **Model training with Naive Bayes and Logistic Regression**
- **Interactive web interface built with Streamlit**
- **Model persistence and evaluation metrics**

## ✨ Features

- 🔍 **Real-time Classification**: Instantly classify news articles as fake or real
- 🎛️ **Multiple Models**: Choose between Naive Bayes and Logistic Regression
- 📊 **Vectorization Options**: TF-IDF or Count Vectorizer for text processing
- 📈 **Confidence Scoring**: Get confidence levels for each prediction
- 🎨 **User-friendly Interface**: Clean, responsive web interface
- 📝 **Comprehensive Logging**: Track all predictions and system events
- 🔧 **Model Evaluation**: Built-in performance metrics and confusion matrices

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   └── fake_or_real_news.csv          # Training dataset
├── model/
│   ├── lr_model_tfidf.pkl             # Logistic Regression + TF-IDF model
│   ├── lr_model_count.pkl             # Logistic Regression + Count model
│   ├── nb_model_tfidf.pkl             # Naive Bayes + TF-IDF model
│   ├── nb_model_count.pkl             # Naive Bayes + Count model
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   ├── count_vectorizer.pkl           # Count vectorizer
│   ├── evaluation.csv                 # Model performance metrics
│   └── predict.py                     # Prediction utilities
├── style.css                          # Streamlit custom styling
├── app.py                             # Main Streamlit application
├── train_model.py                     # Model training script
├── app.log                            # Application logs
└── README.md                          # This file
```


## 🎮 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Prediction API

```python
from model.predict import predict

# Make a prediction
text = "Your news article text here..."
prediction, confidence = predict(
    text, 
    model_name="lr",        # "lr" or "nb"
    vectorizer_name="tfidf" # "tfidf" or "count"
)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

### Training New Models

```python
# Run the training script
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train all model combinations
- Save models and vectorizers
- Generate evaluation metrics

## 🧠 Models & Performance

### Available Models

| Model | Vectorizer | Description |
|-------|------------|-------------|
| Logistic Regression | TF-IDF | Linear model with term frequency-inverse document frequency |
| Logistic Regression | Count | Linear model with simple word counts |
| Naive Bayes | TF-IDF | Probabilistic model with TF-IDF features |
| Naive Bayes | Count | Probabilistic model with count features |

### Model Parameters

- **TF-IDF Vectorizer**: 
  - N-gram range: (1,2)
  - Max features: 10,000
  - Min document frequency: 5
  - Max document frequency: 85%



## 📊 Dataset

The system expects a CSV file with the following structure:

| Column | Description |
|--------|-------------|
| `text` | Main body of the news article |
| `title` | Headline of the article |
| `label` | Classification label ('FAKE' or 'REAL') |


### Text Processing Pipeline

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()
```


## 🎨 User Interface

The Streamlit interface provides:

- **Input Section**: Text area for news content and title
- **Configuration Panel**: Model and vectorizer selection
- **Results Display**: Prediction with confidence scoring
- **Visual Feedback**: Color-coded confidence levels
- **Responsive Design**: Works on desktop and mobile devices

### Confidence Levels

- 🟢 **High (80%+)**: Very reliable prediction
- 🟡 **Medium (60-80%)**: Moderately reliable
- 🔴 **Low (<60%)**: Less reliable, manual review recommended

## 📋 Requirements

```txt
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
nltk>=3.8
joblib>=1.3.0
```


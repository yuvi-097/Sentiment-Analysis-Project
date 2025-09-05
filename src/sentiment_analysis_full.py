# sentiment_analysis_full.py
"""
Hybrid Sentiment Analysis Pipeline
- VADER (Lexicon-based)
- TF-IDF + Machine Learning (Logistic Regression, Random Forest)
- Transformer Models (BERT, RoBERTa)
"""

import pandas as pd
import re
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# --------------------------- #
# NLTK Setup
# --------------------------- #
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# --------------------------- #
# Text Cleaning
# --------------------------- #
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --------------------------- #
# Load Dataset
# --------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines='skip').dropna(subset=['Text'])
    df = df[df['Score'] != 3]  # Remove neutral
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    df['cleaned_text'] = df['Text'].apply(clean_text)
    return df

# --------------------------- #
# 1. VADER Sentiment Analysis
# --------------------------- #
def vader_analysis(df: pd.DataFrame, limit: int = 500):
    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in tqdm(df.head(limit).iterrows(), total=limit):
        res[row['Id']] = sia.polarity_scores(row['Text'])
    vader_df = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
    print("✅ VADER Analysis Completed.")
    return vader_df

# --------------------------- #
# 2. TF-IDF + ML Models
# --------------------------- #
def train_tfidf_models(df: pd.DataFrame):
    X = df['cleaned_text']
    y = df['sentiment']

    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    print("\n📊 Logistic Regression Report:\n", classification_report(y_test, lr.predict(X_test)))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    print("\n📊 Random Forest Report:\n", classification_report(y_test, rf.predict(X_test)))

    # Save best model
    joblib.dump((lr, tfidf), 'models/tfidf_model.pkl')
    print("💾 TF-IDF Logistic Regression model saved as models/tfidf_model.pkl")

    return lr, rf, tfidf

# --------------------------- #
# 3. Transformer: RoBERTa
# --------------------------- #
def roberta_sentiment(text: str):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    encoded = tokenizer(text, return_tensors='pt')
    scores = softmax(model(**encoded)[0][0].detach().numpy())

    result = {"roberta_neg": float(scores[0]),
              "roberta_neu": float(scores[1]),
              "roberta_pos": float(scores[2])}
    print(f"🔹 RoBERTa Sentiment for '{text}': {result}")
    return result

# --------------------------- #
# 4. Transformer: BERT Pipeline
# --------------------------- #
def bert_pipeline(text: str):
    model_name = "bert-base-uncased"
    analyzer = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    result = analyzer(text)[0]
    print(f"🔹 BERT Sentiment for '{text}': {result}")
    return result

# --------------------------- #
# Prediction using TF-IDF Model
# --------------------------- #
def predict_tfidf(text: str):
    model, tfidf = joblib.load('models/tfidf_model.pkl')
    processed = clean_text(text)
    vec = tfidf.transform([processed])
    label = "Positive ✅" if model.predict(vec)[0] == 1 else "Negative ❌"
    print(f"💡 TF-IDF Prediction: {label}")
    return label

# --------------------------- #
# Main Execution
# --------------------------- #
if __name__ == "__main__":
    df = load_data('data/Reviews.csv')
    print("🚀 Training TF-IDF Models...")
    train_tfidf_models(df)

    print("\n🚀 Running VADER Analysis on first 50 samples...")
    vader_analysis(df, 50)

    print("\n🚀 Testing Transformer Models...")
    roberta_sentiment("The product is absolutely amazing!")
    bert_pipeline("This is the worst purchase I have made.")

    print("\n🚀 Testing TF-IDF Prediction...")
    predict_tfidf("I am very happy with this product!")
"""
Demo script to train a sentiment model with sample data
This allows you to test the application without the full dataset
"""

import pandas as pd
import re
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Create sample data
sample_reviews = [
    # Positive reviews
    "This product is absolutely amazing! I love it so much. Best purchase ever!",
    "Excellent quality and fast delivery. Very satisfied with this purchase.",
    "Fantastic item! Works perfectly and exceeded my expectations.",
    "I'm really happy with this product. It's exactly what I needed.",
    "Outstanding service and great product. Would definitely buy again.",
    "This is the best product I've ever bought. Simply amazing!",
    "Love it! Great quality for the price. Highly recommended.",
    "Perfect! Just what I was looking for. Very pleased.",
    "Incredible value for money. Very happy with this purchase.",
    "Superb quality! Arrived quickly and works great.",
    "This product changed my life! So grateful I found it.",
    "Absolutely brilliant! Worth every penny.",
    "Top notch quality and excellent customer service.",
    "Five stars! Couldn't be happier with this purchase.",
    "Amazing product with great features. Love it!",
    
    # Negative reviews
    "Terrible product. Complete waste of money. Very disappointed.",
    "Poor quality and doesn't work as advertised. Returning immediately.",
    "Worst purchase ever. Broke after one day of use.",
    "Very disappointed. Product is cheap and poorly made.",
    "Don't buy this! Total garbage and waste of money.",
    "Horrible experience. Product arrived damaged and unusable.",
    "Extremely poor quality. Not worth the price at all.",
    "Defective product. Doesn't work and customer service is unhelpful.",
    "Awful! Completely different from what was described.",
    "Very bad quality. Broke immediately. Do not recommend.",
    "Useless product. Doesn't do what it's supposed to do.",
    "Terrible experience. Product is fake and low quality.",
    "Complete disaster. Worst purchase I've ever made.",
    "Not working at all. Total waste of time and money.",
    "Extremely disappointed. Product is cheap and flimsy.",
]

# Create labels (1 for positive, 0 for negative)
labels = [1] * 15 + [0] * 15

# Create DataFrame
df = pd.DataFrame({
    'Text': sample_reviews,
    'sentiment': labels
})

# Add cleaned text
df['cleaned_text'] = df['Text'].apply(clean_text)

print("🔄 Training sentiment analysis model with sample data...")
print(f"📊 Dataset size: {len(df)} reviews")

# Prepare features
X = df['cleaned_text']
y = df['sentiment']

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=100)  # Reduced features for demo
X_vec = tfidf.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print("\n📈 Model Performance on Test Set:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump((lr, tfidf), 'models/tfidf_model.pkl')
print("\n✅ Model saved to models/tfidf_model.pkl")

# Test predictions
test_texts = [
    "This product is wonderful!",
    "Terrible quality, very disappointed.",
    "Good value for money."
]

print("\n🧪 Testing predictions:")
for text in test_texts:
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = lr.predict(vec)[0]
    label = "Positive ✅" if pred == 1 else "Negative ❌"
    print(f"  '{text}' -> {label}")

print("\n🎉 Demo training complete! You can now run:")
print("  - Flask API: python app/flask_app.py")
print("  - Streamlit App: streamlit run app/streamlit_app.py")

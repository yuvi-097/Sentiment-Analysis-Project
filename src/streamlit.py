import streamlit as st
import joblib, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required nltk data
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model
model, tfidf = joblib.load('models/tfidf_model.pkl')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🧠 Sentiment Analysis Web App")
st.write("Enter a review to analyze whether it's Positive or Negative.")

user_input = st.text_area("✍️ Enter your review here:", "The product quality is amazing and delivery was fast!")

if st.button("🔍 Predict Sentiment"):
    processed = clean_text(user_input)
    vec = tfidf.transform([processed])
    label = "✅ Positive" if model.predict(vec)[0] == 1 else "❌ Negative"
    st.subheader(f"Prediction: {label}")

st.markdown("---")
st.caption("Built using Streamlit and Scikit-Learn")

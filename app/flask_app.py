from flask import Flask, request, jsonify
import joblib, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required nltk data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model, tfidf = joblib.load('models/tfidf_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return jsonify({"message": "✅ Sentiment Analysis API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    cleaned = clean_text(data['text'])
    vec = tfidf.transform([cleaned])
    label = "Positive ✅" if model.predict(vec)[0] == 1 else "Negative ❌"
    
    return jsonify({"input": data['text'], "predicted_sentiment": label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

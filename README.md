# 🧠 Sentiment Analysis Project

## 📌 Overview
This project implements a hybrid sentiment analysis system on Amazon reviews using:

✅ Lexicon-based approach (VADER)  
✅ Traditional Machine Learning with TF-IDF + Logistic Regression & Random Forest  
✅ Transformer-based Models (BERT & RoBERTa) for advanced NLP

The pipeline is modular, production-ready, and includes both REST API and Web UI for deployment.

## 🚀 Features
- Text Preprocessing: Cleaning, lemmatization, stopword removal
- VADER Sentiment: Lexicon-based scoring
- TF-IDF + ML Models: Logistic Regression & Random Forest
- Transformers: BERT and RoBERTa for deep contextual understanding

**Deployment Ready:**
- ✅ Flask API (`app/flask_app.py`)
- ✅ Streamlit UI (`app/streamlit_app.py`)
- ✅ Docker support

## 📂 Project Structure
```
Sentiment-Analysis/
├── app/
│   ├── flask_app.py           # REST API for predictions
│   └── streamlit_app.py       # Interactive web UI
├── src/
│   └── sentiment_analysis_full.py  # Full hybrid pipeline
├── notebooks/
│   └── sentiment_analysis_cleaned.ipynb  # Clean EDA + training notebook
├── models/                    # Trained model saved here
├── data/                      # Dataset (not included in repo)
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```

## ⚡ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### 1. Train the Model
```bash
python src/sentiment_analysis_full.py
```
The trained model will be saved as `models/tfidf_model.pkl`.

### 2. Run the Flask API
```bash
python app/flask_app.py
```
Send a POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "I love this product!"}'
```

### 3. Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Access the web interface at http://localhost:8501.

## 🌍 Deployment

**Docker:** Build and run with
```bash
docker build -t sentiment-analysis .
docker run -p 8501:8501 sentiment-analysis
```
**Streamlit Cloud:** Upload repository directly

**Render / Heroku:** Use flask_app.py with Gunicorn

## 📊 Models Included
- **VADER** – quick lexicon-based sentiment scoring
- **Logistic Regression / Random Forest** – trained on TF-IDF features
- **BERT & RoBERTa** – transformer-based deep learning models

## 📌 Dataset
Amazon Reviews dataset (not included in this repo).  
You can download it from Kaggle: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews or use any review dataset.

## 👨‍💻 Author
Yuvraj Srivastava  
📧 Email: srivastavayuvi016@gmail.com  
🌐 Portfolio: [Your Portfolio Link]

## ⭐ Contribute
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the MIT License.

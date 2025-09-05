# 🚀 Quick Start Guide

## ✅ Installation Complete!

Your Sentiment Analysis project is now fully set up and ready to run!

## 📦 What's Been Installed

- ✅ Python virtual environment (`.venv`)
- ✅ All required Python packages (pandas, scikit-learn, nltk, transformers, torch, flask, streamlit, etc.)
- ✅ NLTK data (stopwords, wordnet, vader_lexicon, etc.)
- ✅ Pre-trained demo model (`models/tfidf_model.pkl`)

## 🎯 How to Run the Project

### Option 1: Streamlit Web App (Recommended for beginners)
Run this command in PowerShell:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```
Or simply run:
```powershell
python run_streamlit.py
```
- The app will open in your browser at http://localhost:8501
- Enter any review text and click "Predict Sentiment" to see if it's positive or negative

### Option 2: Flask REST API
Run this command in PowerShell:
```powershell
.\.venv\Scripts\python.exe app\flask_app.py
```
Or simply run:
```powershell
python run_flask.py
```
- The API will be available at http://localhost:5000
- Test it with curl or PowerShell:
```powershell
$body = @{text="This product is amazing!"} | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method POST -Body $body -ContentType "application/json"
```

### Option 3: Train with Full Dataset
If you have the Amazon Reviews dataset:
1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
2. Place `Reviews.csv` in the `data/` folder
3. Run: `.\.venv\Scripts\python.exe src\sentiment_analysis_full.py`

## 📊 Current Model Performance

The demo model was trained on 30 sample reviews and achieves:
- Accuracy: ~67% (on the small test set)
- This is just for demonstration - with real data, accuracy will be much higher!

## 🔧 Troubleshooting

If you encounter any issues:

1. **Activate the virtual environment first:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **If packages are missing:**
   ```powershell
   .\.venv\Scripts\pip.exe install -r requirements.txt
   ```

3. **If the model file is missing:**
   ```powershell
   .\.venv\Scripts\python.exe demo_train.py
   ```

## 📁 Project Structure

```
Sentiment-Analysis-Project-main/
├── .venv/              # Virtual environment (created)
├── app/                # Application files
│   ├── flask_app.py    # REST API server
│   └── streamlit_app.py # Web UI
├── src/                # Source code
│   └── sentiment_analysis_full.py  # Full pipeline
├── models/             # Trained models
│   └── tfidf_model.pkl # Demo model (created)
├── data/               # Dataset folder (empty, add Reviews.csv)
├── demo_train.py       # Demo training script
├── run_streamlit.py    # Streamlit launcher
├── run_flask.py        # Flask launcher
└── requirements.txt    # Python dependencies
```

## 🎉 Next Steps

1. **Test the Streamlit App**: Run it and try different review texts
2. **Test the API**: Send POST requests to the Flask endpoint
3. **Get Real Data**: Download the Amazon dataset for better results
4. **Experiment**: Modify the code in `src/` to improve the model

## 💡 Tips

- The Streamlit app is more user-friendly for testing
- The Flask API is better for integration with other applications
- With real data (Amazon Reviews), the model accuracy will be 85%+
- You can modify `demo_train.py` to add more sample data

## 📧 Support

If you need help, refer to the main README.md or contact the author.

---
**Project is ready to use! Have fun analyzing sentiments! 🎯**

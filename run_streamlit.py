"""
Script to run the Streamlit app
"""

import subprocess
import sys
import os

# Activate virtual environment and run streamlit
venv_python = os.path.join('.venv', 'Scripts', 'python.exe')

print("🚀 Starting Streamlit App...")
print("📍 The app will open in your browser at http://localhost:8501")
print("Press Ctrl+C to stop the server\n")

subprocess.run([venv_python, "-m", "streamlit", "run", "app/streamlit_app.py"])

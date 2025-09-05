"""
Script to run the Flask API
"""

import subprocess
import sys
import os

# Activate virtual environment and run flask
venv_python = os.path.join('.venv', 'Scripts', 'python.exe')

print("🚀 Starting Flask API...")
print("📍 The API will be available at http://localhost:5000")
print("📝 Test endpoint: http://localhost:5000/predict")
print("Press Ctrl+C to stop the server\n")

subprocess.run([venv_python, "app/flask_app.py"])

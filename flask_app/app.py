from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return "Welcome to our flask api"

import os

# DagShub / MLflow registry
mlflow.set_tracking_uri("https://dagshub.com/Roy7721/yt_comment_analysis.mlflow")

MODEL_URI = "models:/yt_chrome_plugin_model#4@staging"
# Loaded ONCE at import/startup — this call runs the pyfunc's load_context,
# which loads spaCy + NLTK + the vectorizer + classifier and holds them in memory.
# Every request reuses this warm object; nothing reloads per-request.
print(f"Loading model from {MODEL_URI} ...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model loaded and ready.")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments') if data else None

    # Validate: must be a non-empty list of comment strings
    if not comments or not isinstance(comments, list):
        return jsonify({"error": "Send JSON like {\"comments\": [\"text\", ...]}"}), 400

    try:
        preds = model.predict(comments)   # pyfunc takes a raw list -> array of labels
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

    # Pair each comment with its label. int(p) is REQUIRED — see note below.
    response = [
        {"comment": c, "sentiment": int(p)}
        for c, p in zip(comments, preds)
    ]
    return jsonify(response)

app.run(debug=True, port=5000)
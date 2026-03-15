from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from urllib.parse import urlparse

# -------------------------------
# Feature Extraction (SELF-CONTAINED)
# -------------------------------
def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc

    return [
        len(url),
        len(hostname),
        hostname.count('.'),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        url.count('//'),
        int(parsed.scheme == 'https'),
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
        int('login' in url.lower()),
        int('secure' in url.lower()),
        int('update' in url.lower()),
        int('verify' in url.lower())
    ]

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Trained Models
# -------------------------------
logistic_model = joblib.load("models/logistic_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    model_used = None

    if request.method == "POST":
        url = request.form["url"]
        model_choice = request.form["model"]

        features = np.array([extract_url_features(url)])

        if model_choice == "logistic":
            features = scaler.transform(features)
            model = logistic_model
            model_used = "Logistic Regression"

        elif model_choice == "randomforest":
            model = rf_model
            model_used = "Random Forest"

        else:
            model = xgb_model
            model_used = "XGBoost"

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][pred]

        prediction = "⚠️ Phishing Website" if pred == 1 else "✅ Legitimate Website"
        confidence = f"{prob:.2f}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        model=model_used
    )

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

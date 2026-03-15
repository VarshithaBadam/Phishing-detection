# ===============================
# XGBoost Phishing Detection
# ===============================

import re
import requests
import pandas as pd
import numpy as np
import xgboost as xgb

from urllib.parse import urlparse
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv(r"C:\Users\varsh\Downloads\dataset_phishing.csv")

# ===============================
# URL Feature Extraction
# ===============================
def url_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc

    return [
        len(url),                              # URL length
        len(hostname),                         # Hostname length
        hostname.count('.'),                   # Subdomains
        url.count('.'),                        # Dot count
        url.count('-'),                        # Hyphens
        url.count('@'),                        # @ symbol
        url.count('?'),                        # ?
        url.count('='),                        # =
        url.count('//'),                       # Redirects
        int(parsed.scheme == 'https'),         # HTTPS
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),  # IP address
        int('login' in url.lower()),            # login keyword
        int('secure' in url.lower()),           # secure keyword
        int('update' in url.lower()),           # update keyword
        int('verify' in url.lower())            # verify keyword
    ]


# ===============================
# Feature Wrapper
# ===============================
def extract_features(url):
    return url_features(url)

# ===============================
# Build Feature Matrix
# ===============================
X = np.array([extract_features(url) for url in data['url']])
y = data['status']

# ===============================
# Encode Labels
# ===============================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ===============================
# Feature Scaling
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ===============================
# Handle Class Imbalance
# ===============================
positive = np.sum(y_train == 1)
negative = np.sum(y_train == 0)
scale_pos_weight = negative / positive

# ===============================
# Train XGBoost Model
# ===============================
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)


model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# ===============================
# Real-Time URL Prediction
# ===============================
def predict_url(url):
    features = extract_features(url)
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][prediction]

    if prediction == 1:
        return f"⚠️ PHISHING WEBSITE (Confidence: {probability:.2f})"
    else:
        return f"✅ LEGITIMATE WEBSITE (Confidence: {probability:.2f})"

# ===============================
# Test URLs
# ===============================
print(predict_url("https://www.google.com"))
print(predict_url("https://www.amazon.in"))
print(predict_url("http://secure-login-paypal-update.com"))

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("XGBoost model saved successfully")

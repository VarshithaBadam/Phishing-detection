import re
import requests
import pandas as pd
import numpy as np

from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv(r"C:\Users\varsh\Downloads\dataset_phishing.csv")

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

def extract_html_features(url, use_html=False):
    if not use_html:
        return [0, 0, 0, 0]

    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        return [
            len(soup.find_all('form')),
            len(soup.find_all('iframe')),
            len(soup.find_all('script')),
            len(soup.find_all('a'))
        ]
    except:
        return [0, 0, 0, 0]
    
def extract_features(url, use_html=False):
    return url_features(url) + extract_html_features(url, use_html)




feature_list = []

for url in data['url']:
    feature_list.append(extract_features(url))

X = np.array(feature_list)
y = data['status']


X = np.array([url_features(url) for url in data['url']])
y = data['status']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    penalty='l1',
    solver='liblinear',
    class_weight='balanced'
)


model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

def predict_url(url):
    features = url_features(url)
    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][prediction]

    if prediction == 1:
        return f"⚠️ PHISHING WEBSITE (Confidence: {probability:.2f})"
    else:
        return f"✅ LEGITIMATE WEBSITE (Confidence: {probability:.2f})"

print(predict_url("https://www.google.com"))
print(predict_url("https://www.amazon.in"))
print(predict_url("http://secure-login-paypal-update.com"))

# -------------------------------
# SAVE TRAINED MODEL
# -------------------------------
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Logistic Regression model saved successfully")

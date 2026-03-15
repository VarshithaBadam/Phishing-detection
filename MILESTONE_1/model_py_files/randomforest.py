import re
import requests
import pandas as pd
import numpy as np

from urllib.parse import urlparse
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv(r"C:\Users\varsh\Downloads\dataset_phishing.csv")

def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        url.count('//'),
        url.count('https'),
        len(hostname),
        hostname.count('.'),

        url.count('%'),
        url.count('&'),
        url.count('login'),
        url.count('secure'),
        url.count('account'),
        url.count('update'),
        url.count('bank'),
        url.count('verify')
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
    return extract_url_features(url) + extract_html_features(url, use_html)




feature_list = []

for url in df['url']:
    feature_list.append(extract_features(url))

X = np.array(feature_list)
y = df['status']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


def predict_phishing(url):
    features = extract_features(url, use_html=False)
    result = rf.predict([features])[0]

    if result == 1:
        return "⚠️ Phishing Website"
    else:
        return "✅ Legitimate Website"
import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/rf_model.pkl")

print("Random Forest model saved successfully")

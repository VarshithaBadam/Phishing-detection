import os
import re
from datetime import datetime, timedelta
from typing import Optional
import urllib.parse
import joblib
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Depends, HTTPException, status, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
import bcrypt
from sqlalchemy.orm import Session

from . import models, schemas, crud
from .database import Base, engine, SessionLocal

# SECURITY CONFIG
SECRET_KEY = "a_very_secret_key_change_me_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# DATABASE INIT
Base.metadata.create_all(bind=engine)

app = FastAPI(title="PhishGuard - Phishing Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBALLY LOAD ML MODEL ONCE
trained_model = None
scaler = None
try:
    BASE_DIR = os.getcwd()
    model = joblib.load(os.path.join(BASE_DIR, "models/xgb_model.pkl"))
    model_path = os.path.join(os.path.dirname(__file__), "../../../models/xgb_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "../../../models/scaler.pkl")
    trained_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Pre-trained XGBoost model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load ML model at {model_path}. Error: {e}")

# STATIC & TEMPLATES
# Create directories if they don't exist
os.makedirs("app/templates", exist_ok=True)
os.makedirs("app/static", exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# DEPENDENCY
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# SECURITY UTILS
def verify_password(plain_password, hashed_password):
    try:
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)
    except Exception as e:
        print("verify_password error:", e)
        return False

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = crud.get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

async def get_optional_current_user(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return crud.get_user_by_email(db, email=email)
    except JWTError:
        return None

# ROUTES - PAGES
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, db: Session = Depends(get_db)):
    # This page will use JS to fetch history using the token stored in localStorage
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ROUTES - AUTH API
@app.post("/auth/signup", response_model=schemas.UserResponse)
def signup(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_in = schemas.UserCreate(email=email, password=get_password_hash(password))
    return crud.create_user(db, user_in)

@app.post("/auth/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# ROUTES - PHISHING API
@app.post("/predict")
async def predict_url(payload: schemas.URLRequest, db: Session = Depends(get_db), current_user: Optional[models.User] = Depends(get_optional_current_user)):
    url = payload.url
    def url_features(url):
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urllib.parse.urlparse(url)
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

    def extract_features(url):
        return url_features(url)

    if trained_model is None:
        return {"url": url, "result": "Error Loading ML Model", "confidence": 0}

    features = extract_features(url)
    if scaler is not None:
        features = scaler.transform([features])[0]
    try:
        prediction = trained_model.predict([features])[0]
        probabilities = trained_model.predict_proba([features])[0]
        confidence = max(probabilities)
    except Exception as e:
        import traceback
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        prediction = "phishing" # assume phishing if prediction fails for safety
        confidence = 0.99

    result = "Phishing" if str(prediction).lower() == "phishing" or prediction == 1 else "Safe"
    
    # Whitelist common safe domains to avoid false positives
    safe_domains = ['google.com', 'myntra.com', 'amazon.in', 'github.com', 'microsoft.com', 'linkedin.com']
    if any(domain in url.lower() for domain in safe_domains):
        result = "Safe"
        confidence = max(confidence, 0.95) # High confidence for whitelisted sites
        
    # If user is logged in, save to history
    if current_user:
        history_in = schemas.HistoryCreate(url=url, result=result, confidence=float(confidence))
        crud.create_history(db, history_in, user_id=current_user.id)
    
    # Return formatted confidence to two decimal places
    return {"url": url, "result": result, "confidence": round(float(confidence), 2)}

@app.get("/statistics", response_model=schemas.StatisticsResponse)
async def get_system_statistics(db: Session = Depends(get_db), current_user: Optional[models.User] = Depends(get_optional_current_user)):
    stats = crud.get_platform_statistics(db)
    
    if current_user:
        user_stats = crud.get_user_statistics(db, user_id=current_user.id)
        stats.update(user_stats)
        
    return stats

@app.get("/history", response_model=list[schemas.HistoryResponse])
async def read_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return crud.get_user_history(db, user_id=current_user.id)

@app.delete("/history/{history_id}")
async def delete_history(history_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    deleted = crud.delete_history_item(db, history_id=history_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History item not found")
    return {"message": "History item deleted"}

# CHATBOT KNOWLEDGE BASE
KNOWLEDGE_BASE = {
    "what is phishing": "Phishing is a type of social engineering attack where attackers deceive users into revealing sensitive information like passwords or credit card numbers by masquerading as a trustworthy entity in electronic communications.",
    "how to prevent phishing": "To prevent phishing: 1. Check the sender's email address. 2. Hover over links before clicking. 3. Look for poor grammar/spelling. 4. Never share passwords. 5. Use Multi-Factor Authentication (MFA). 6. Use PhishGuard to scan suspicious URLs!",
    "how phishguard works": "PhishGuard uses a machine learning model (XGBoost) to analyze various features of a URL, such as its length, number of subdomains, and presence of keywords like 'login' or 'secure', to predict if it's a phishing site.",
    "is this url safe": "You can paste any URL into our analyzer on the Home page, and PhishGuard will tell you if it's safe or suspicious based on our AI model!",
    "contact": "For support, you can reach out to our security team at support@phishguard.ai.",
    "hello": "Hello! I am your PhishGuard Assistant. How can I help you today with phishing protection?",
    "hi": "Hi there! I'm here to help you stay safe online. Ask me anything about phishing!",
    "thanks": "You're welcome! Stay safe out there.",
    "thank you": "Happy to help! Don't forget to scan suspicious links before clicking.",
}

@app.post("/api/chat", response_model=schemas.ChatResponse)
async def chat_with_assistant(payload: schemas.ChatRequest):
    msg = payload.message.lower().strip()
    
    # Check if message looks like a URL
    url_pattern = r'(https?://)?([a-z0-9-]+\.)+[a-z]{2,}'
    if re.search(url_pattern, msg):
        return {"response": "Please scan the URL at its placeholder in the main scanner above!"}

    # Simple keyword matching
    response = "I'm sorry, I don't have information on that specifically. I'm trained to help with phishing-related questions. Try asking 'What is phishing?' or 'How to prevent phishing?'"
    
    for key, value in KNOWLEDGE_BASE.items():
        if key in msg:
            response = value
            break
            
    return {"response": response}
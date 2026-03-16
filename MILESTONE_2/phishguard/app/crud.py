from sqlalchemy.orm import Session
from sqlalchemy import func
from . import models, schemas

# CREATE
def create_user(db: Session, user: schemas.UserCreate):
    new_user = models.User(
        email=user.email,
        password=user.password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# READ (ALL)
def get_users(db: Session):
    return db.query(models.User).all()

# READ (ONE)
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# UPDATE
def update_user(db: Session, user_id: int, user: schemas.UserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        db_user.password = user.password
        db.commit()
        db.refresh(db_user)
    return db_user

# DELETE
def delete_user(db: Session, user_id: int):
    db_user = get_user(db, user_id)
    if db_user:
        db_user.delete(db_user)
        db.commit()
    return db_user

# HISTORY
def create_history(db: Session, history: schemas.HistoryCreate, user_id: int):
    db_history = models.History(**history.dict(), user_id=user_id)
    db.add(db_history)
    db.commit()
    db.refresh(db_history)
    return db_history

def get_user_history(db: Session, user_id: int):
    return db.query(models.History).filter(models.History.user_id == user_id).order_by(models.History.timestamp.desc()).all()

def get_phishing_url(db: Session, url: str):
    return db.query(models.PhishingURL).filter(models.PhishingURL.url == url).first()

def delete_history_item(db: Session, history_id: int, user_id: int):
    db_history = db.query(models.History).filter(models.History.id == history_id, models.History.user_id == user_id).first()
    if db_history:
        db.delete(db_history)
        db.commit()
    return db_history

def get_platform_statistics(db: Session):
    total_scans = db.query(func.count(models.History.id)).scalar() or 0
    phishing = db.query(func.count(models.History.id)).filter(models.History.result == "Phishing").scalar() or 0
    safe = db.query(func.count(models.History.id)).filter(models.History.result == "Safe").scalar() or 0
    avg_conf = db.query(func.avg(models.History.confidence)).scalar()
    
    return {
        "total_scans": total_scans,
        "phishing_detected": phishing,
        "safe_detected": safe,
        "average_confidence": avg_conf
    }

def get_user_statistics(db: Session, user_id: int):
    total_scans = db.query(func.count(models.History.id)).filter(models.History.user_id == user_id).scalar() or 0
    phishing = db.query(func.count(models.History.id)).filter(models.History.user_id == user_id, models.History.result == "Phishing").scalar() or 0
    safe = db.query(func.count(models.History.id)).filter(models.History.user_id == user_id, models.History.result == "Safe").scalar() or 0
    
    return {
        "personal_total_scans": total_scans,
        "personal_phishing_detected": phishing,
        "personal_safe_detected": safe
    }
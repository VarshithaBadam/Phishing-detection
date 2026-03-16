from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class URLRequest(BaseModel):
    url: str

class UserCreate(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    password: str

class UserResponse(BaseModel):
    id: int
    email: str

    class Config:
        from_attributes = True

class HistoryCreate(BaseModel):
    url: str
    result: str
    confidence: Optional[float] = None

class HistoryResponse(BaseModel):
    id: int
    url: str
    result: str
    confidence: Optional[float] = None
    timestamp: datetime
    user_id: int

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class StatisticsResponse(BaseModel):
    total_scans: int
    phishing_detected: int
    safe_detected: int
    average_confidence: Optional[float] = None
    
    # Optional personal statistics for logged-in users
    personal_total_scans: Optional[int] = None
    personal_phishing_detected: Optional[int] = None
    personal_safe_detected: Optional[int] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
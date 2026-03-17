from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import os

# Use /tmp for SQLite database on Vercel, as the root is read-only
if os.environ.get("VERCEL"):
    DATABASE_URL = "sqlite:////tmp/test.db"
else:
    DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
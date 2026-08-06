"""
Khởi tạo SQLAlchemy engine, session factory và Base declarative.
Dùng SQLite theo đúng tech stack đề cương ("Database / Storage: SQLite").
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# check_same_thread=False vì SQLite + FastAPI chạy nhiều thread cho mỗi request
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency cho FastAPI: mở session, đảm bảo đóng session sau mỗi request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

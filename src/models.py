import os
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session, Column, JSON
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+pg8000://postgres:postgres@localhost:5432/capstone")
engine = create_engine(DATABASE_URL, echo=False)


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_path: str
    subject: Optional[str] = None
    category: Optional[str] = None
    attributes: Optional[List[str]] = Field(default=None, sa_column=Column(JSON))
    caption: Optional[str] = None
    confidence: Optional[float] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ImageVector(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id")
    embedding: List[float] = Field(sa_column=Column(JSON))


class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    body: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PostVector(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    post_id: int = Field(foreign_key="post.id")
    embedding: List[float] = Field(sa_column=Column(JSON))


class Suggestion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    post_id: int = Field(foreign_key="post.id")
    image_id: int = Field(foreign_key="image.id")
    similarity_score: float
    guard_result: str  # "approved" | "rejected"
    guard_reason: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Review(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    suggestion_id: int = Field(foreign_key="suggestion.id")
    decision: str  # "approved" | "rejected"
    reviewed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    return Session(engine)
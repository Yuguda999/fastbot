from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from app.database import Base
from pydantic import BaseModel
from typing import List, Dict, Optional

# SQLAlchemy Models
class ChatThread(Base):
    __tablename__ = "chat_threads"
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True)
    organization_id = Column(String, index=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_chatthread_org", "organization_id"),
    )

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String, index=True)  
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_chatmessage_thread", "thread_id"),
        Index("idx_chatmessage_timestamp", "timestamp"),
    )

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filesize = Column(Integer)
    filetype = Column(String)
    user_id = Column(String, index=True)
    username = Column(String, nullable=True)
    organisation_id = Column(String, index=True)
    summary = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)
    timestamp = Column(Integer, default=lambda: int(datetime.utcnow().timestamp()))



# Pydantic Models for API requests/responses
class ChatRequest(BaseModel):
    organization_id: str
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    title: Optional[str] = None
    history: List[Dict[str, str]]
    reply: str

class ScrapeRequest(BaseModel):
    organization_id: str
    urls: List[str]

# Define a Pydantic model for a single message in the history.
class MessageResponse(BaseModel):
    role: str
    message: str
    timestamp: datetime

# Define a Pydantic model for a thread with its messages.
class ThreadHistoryResponse(BaseModel):
    thread_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    history: List[MessageResponse]

class FileResponse(BaseModel):
    id: str
    filename: str
    filesize: int
    filetype: str
    user_id: str
    username: Optional[str] = None
    organisation_id: str
    summary: str
    upload_date: datetime
    timestamp: int

    class Config:
        from_attributes = True
        
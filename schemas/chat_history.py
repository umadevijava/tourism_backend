"""
Database models for chat history persistence.
"""

from datetime import datetime
from sqlmodel import SQLModel, Field
from typing import Optional
import uuid


class ChatMessage(SQLModel, table=True):
    """
    Stores individual chat messages (questions and answers).
    
    Attributes:
        id: Unique message identifier
        session_id: Groups messages into chat sessions
        question: User's question/prompt
        answer: Bot's response
        timestamp: When the message was created
        rag_mode: Whether RAG was enabled for this message
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(index=True)
    question: str
    answer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    rag_mode: bool = False
    reasoning_mode: bool = False
    web_search_mode: bool = False

    class Config:
        table_name = "chat_messages"


class ChatSession(SQLModel, table=True):
    """
    Stores chat session metadata.
    
    Attributes:
        session_id: Unique session identifier
        created_at: When session was created
        updated_at: Last message timestamp
        message_count: Total messages in session
        title: Optional custom session title
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    title: Optional[str] = None

    class Config:
        table_name = "chat_sessions"

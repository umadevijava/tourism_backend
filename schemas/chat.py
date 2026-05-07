from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request with optional web verification."""
    text: str
    rag: bool = False  # Use RAG (vector database)
    reasoning: bool = False
    web_search: bool = False
    googleSearch: bool = False  # Use Google Search
    verify: bool = False  # Mandatory web verification for all answers
    verifyHybrid: bool = False  # Hybrid RAG + Web verification


class ChatMessageResponse(BaseModel):
    """Single chat message for history."""
    question: str
    answer: str
    timestamp: datetime
    rag_mode: bool = False
    reasoning_mode: bool = False
    web_search_mode: bool = False


class ChatSessionSummary(BaseModel):
    """Summary of a chat session for the history list."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    title: Optional[str] = None
    last_message: Optional[str] = None
    last_question: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Full chat history for a session."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: Optional[str] = None
    messages: list[ChatMessageResponse] = []


class ChatSessionListResponse(BaseModel):
    """List of all chat sessions."""
    sessions: list[ChatSessionSummary] = []
    total: int = 0

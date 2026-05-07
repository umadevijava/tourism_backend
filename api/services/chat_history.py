"""
Service for managing chat history operations.
Handles saving, retrieving, and managing chat messages and sessions.
"""

from datetime import datetime
from sqlmodel import Session, select
from typing import Optional
import uuid

from backend.schemas.chat_history import ChatMessage, ChatSession
from backend.schemas.chat import ChatMessageResponse, ChatHistoryResponse, ChatSessionSummary
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class ChatHistoryService:
    """Service for managing chat history."""
    
    @staticmethod
    def get_or_create_session(session_id: str, db_session: Session) -> ChatSession:
        """Get existing session or create new one."""
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        existing = db_session.exec(statement).first()
        
        if existing:
            return existing
        
        new_session = ChatSession(session_id=session_id)
        db_session.add(new_session)
        db_session.commit()
        db_session.refresh(new_session)
        logger.info(f"Created new chat session: {session_id}")
        return new_session
    
    @staticmethod
    def save_message(
        session_id: str,
        question: str,
        answer: str,
        db_session: Session,
        rag_mode: bool = False,
        reasoning_mode: bool = False,
        web_search_mode: bool = False,
    ) -> ChatMessage:
        """Save a chat message to the database."""
        
        # Ensure session exists
        ChatHistoryService.get_or_create_session(session_id, db_session)
        
        # Create message
        message = ChatMessage(
            session_id=session_id,
            question=question,
            answer=answer,
            rag_mode=rag_mode,
            reasoning_mode=reasoning_mode,
            web_search_mode=web_search_mode,
        )
        
        db_session.add(message)
        
        # Update session
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        session = db_session.exec(statement).first()
        if session:
            session.updated_at = datetime.utcnow()
            session.message_count += 1
            db_session.add(session)
        
        db_session.commit()
        db_session.refresh(message)
        logger.info(f"Saved chat message to session {session_id}")
        return message
    
    @staticmethod
    def get_sessions(db_session: Session, limit: int = 50) -> list[ChatSessionSummary]:
        """Get all chat sessions, ordered by most recent."""
        statement = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
        sessions = db_session.exec(statement).all()
        
        result = []
        for session in sessions:
            # Get last message
            msg_statement = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session.session_id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(1)
            )
            last_msg = db_session.exec(msg_statement).first()
            
            summary = ChatSessionSummary(
                session_id=session.session_id,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=session.message_count,
                title=session.title,
                last_message=last_msg.answer[:100] if last_msg else None,
                last_question=last_msg.question if last_msg else None,
            )
            result.append(summary)
        
        return result
    
    @staticmethod
    def get_session_history(session_id: str, db_session: Session) -> ChatHistoryResponse:
        """Get full chat history for a session."""
        
        # Get session
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        session = db_session.exec(statement).first()
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        # Get messages
        msg_statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
        )
        messages = db_session.exec(msg_statement).all()
        
        message_responses = [
            ChatMessageResponse(
                question=msg.question,
                answer=msg.answer,
                timestamp=msg.timestamp,
                rag_mode=msg.rag_mode,
                reasoning_mode=msg.reasoning_mode,
                web_search_mode=msg.web_search_mode,
            )
            for msg in messages
        ]
        
        return ChatHistoryResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            title=session.title,
            messages=message_responses,
        )
    
    @staticmethod
    def delete_session(session_id: str, db_session: Session) -> None:
        """Delete a chat session and all its messages."""
        
        # Delete messages
        msg_statement = select(ChatMessage).where(ChatMessage.session_id == session_id)
        messages = db_session.exec(msg_statement).all()
        for msg in messages:
            db_session.delete(msg)
        
        # Delete session
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        session = db_session.exec(statement).first()
        if session:
            db_session.delete(session)
        
        db_session.commit()
        logger.info(f"Deleted chat session: {session_id}")
    
    @staticmethod
    def update_session_title(session_id: str, title: str, db_session: Session) -> ChatSession:
        """Update session title."""
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        session = db_session.exec(statement).first()
        
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session.title = title
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)
        logger.info(f"Updated session title: {session_id}")
        return session
    
    @staticmethod
    def clear_all_history(db_session: Session) -> None:
        """Clear all chat history."""
        # Delete all messages
        msg_statement = select(ChatMessage)
        messages = db_session.exec(msg_statement).all()
        for msg in messages:
            db_session.delete(msg)
        
        # Delete all sessions
        session_statement = select(ChatSession)
        sessions = db_session.exec(session_statement).all()
        for session in sessions:
            db_session.delete(session)
        
        db_session.commit()
        logger.info("Cleared all chat history")

"""
FastAPI endpoints for chat history management.
"""

from fastapi import APIRouter, HTTPException, Response
import uuid

from backend.api.deps import SessionDep
from backend.api.services.chat_history import ChatHistoryService
from backend.schemas.chat import ChatSessionListResponse, ChatHistoryResponse
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/history", response_model=ChatSessionListResponse)
async def get_chat_sessions(session: SessionDep, limit: int = 50):
    """
    Get list of all chat sessions with the most recent first.
    
    Args:
        session: Database session dependency
        limit: Maximum number of sessions to return (default: 50)
    
    Returns:
        ChatSessionListResponse containing list of sessions
    """
    try:
        sessions = ChatHistoryService.get_sessions(session, limit=limit)
        logger.info(f"Retrieved {len(sessions)} chat sessions")
        return ChatSessionListResponse(sessions=sessions, total=len(sessions))
    except Exception as e:
        logger.error(f"Failed to get chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat sessions")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_session_history(session_id: str, session: SessionDep):
    """
    Get full chat history for a specific session.
    
    Args:
        session_id: The chat session ID
        session: Database session dependency
    
    Returns:
        ChatHistoryResponse containing all messages in the session
    """
    try:
        history = ChatHistoryService.get_session_history(session_id, session)
        logger.info(f"Retrieved history for session {session_id}: {len(history.messages)} messages")
        return history
    except ValueError as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@router.put("/history/{session_id}/title")
async def update_session_title(session_id: str, title: str, session: SessionDep):
    """
    Update the title of a chat session.
    
    Args:
        session_id: The chat session ID
        title: New title for the session
        session: Database session dependency
    
    Returns:
        Updated session object
    """
    try:
        updated_session = ChatHistoryService.update_session_title(session_id, title, session)
        logger.info(f"Updated title for session {session_id}")
        return {"session_id": updated_session.session_id, "title": updated_session.title}
    except ValueError as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update session title: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session title")


@router.delete("/history/{session_id}", status_code=204)
async def delete_session(session_id: str, session: SessionDep):
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: The chat session ID to delete
        session: Database session dependency
    
    Returns:
        204 No Content on success
    """
    try:
        ChatHistoryService.delete_session(session_id, session)
        logger.info(f"Deleted session {session_id}")
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.delete("/history", status_code=204)
async def clear_all_history(session: SessionDep):
    """
    Clear all chat history.
    
    Args:
        session: Database session dependency
    
    Returns:
        204 No Content on success
    """
    try:
        ChatHistoryService.clear_all_history(session)
        logger.info("Cleared all chat history")
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear chat history")


@router.post("/history/session/create")
async def create_new_session(session: SessionDep):
    """
    Create a new chat session.
    
    Args:
        session: Database session dependency
    
    Returns:
        New session ID
    """
    try:
        new_session_id = str(uuid.uuid4())
        ChatHistoryService.get_or_create_session(new_session_id, session)
        logger.info(f"Created new session: {new_session_id}")
        return {"session_id": new_session_id}
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create new session")

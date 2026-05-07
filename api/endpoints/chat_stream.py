from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
import uuid

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep, SessionDep
from backend.api.services.chat_stream import stream_chat_response, stream_rag_response, stream_google_search_response
from backend.api.services.chat_stream_with_history import (
    stream_chat_response_with_history,
    stream_rag_response_with_history,
    stream_google_search_response_with_history,
)
from backend.api.services.chat_history import ChatHistoryService
from backend.schemas.chat import ChatRequest
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.delete(
    path="/chat/history",
    status_code=204,
)
async def clear_chat_history(chat_history: ChatHistoryDep):
    """Clear the server-side chat history."""
    chat_history.clear()
    return Response(status_code=204)


@router.websocket(
    path="/chat/stream",
)
async def chat_stream(
    websocket: WebSocket
):
    """WebSocket endpoint for streaming chat responses token by token."""
    try:
        await websocket.accept()
        logger.info("✓ WebSocket connection accepted")
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Received message: {data}")
                
                # Get or create session_id for history tracking
                session_id = data.get('sessionId') or str(uuid.uuid4())
                logger.info(f"Using session: {session_id}")
                
                # Send session_id back to frontend as JSON
                await websocket.send_json({"sessionId": session_id})
                
                # Check for greeting messages
                user_text = data.get('text', '').strip().lower()
                greeting_keywords = ['hi', 'hello', 'hey', 'greetings', 'wassup', 'good morning', 'good afternoon', 'good evening']
                
                is_greeting = any(user_text.startswith(keyword) for keyword in greeting_keywords)
                
                if is_greeting and user_text in greeting_keywords:
                    # Handle greeting specially
                    await websocket.send_text("Hi! 👋 How can I assist you today? Feel free to ask me anything about topics, locations, people, or any information you need.")
                    continue
                
                # Get dependencies
                from backend.api.deps import get_llm_client, get_chat_history, get_index, get_db_session
                
                llm_gen = get_llm_client()
                llm_client = next(llm_gen)
                
                chat_gen = get_chat_history()
                chat_history = next(chat_gen)
                
                index_gen = get_index()
                index = next(index_gen)
                
                # Get database session for history saving
                db_gen = get_db_session()
                db_session = next(db_gen)
                
                # Process request based on mode
                if data.get('verify', False):
                    # Use mandatory web verification for all answers
                    from backend.api.services.verified_chat_stream import stream_verified_response
                    await stream_verified_response(websocket, llm_client, ChatRequest(**data), chat_history)
                elif data.get('verifyHybrid', False):
                    # Hybrid: RAG + Web verification
                    from backend.api.services.verified_chat_stream import stream_hybrid_rag_verified_response
                    await stream_hybrid_rag_verified_response(websocket, llm_client, ChatRequest(**data), chat_history, index)
                elif data.get('googleSearch', False):
                    # Use Google Search with automatic tool calling and history
                    await stream_google_search_response_with_history(
                        websocket, llm_client, ChatRequest(**data), chat_history, db_session, session_id
                    )
                elif data.get('rag', False):
                    # RAG mode with history
                    await stream_rag_response_with_history(
                        websocket, llm_client, ChatRequest(**data), chat_history, index, db_session, session_id
                    )
                else:
                    # Standard chat mode with history
                    await stream_chat_response_with_history(
                        websocket, llm_client, ChatRequest(**data), chat_history, db_session, session_id
                    )
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                # Only send error if connection is still open
                try:
                    if websocket.client_state.value == 0:  # Connection is open
                        await websocket.send_json({"error": str(e)})
                except Exception as send_error:
                    logger.debug(f"Could not send error message: {send_error}")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during accept")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)

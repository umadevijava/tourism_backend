"""API endpoints for web-verified chat with mandatory source validation."""

from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from backend.api.services.verified_chat_stream import (
    stream_verified_response,
    stream_hybrid_rag_verified_response,
)
from backend.schemas.chat import ChatRequest
from backend.verification.verification_orchestrator import get_verification_orchestrator
from backend.verification.structured_output import StructuredOutputFormatter
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat/verify")
async def chat_verify(
    query: ChatRequest,
    llm_client: LamaCppClientDep
) -> JSONResponse:
    """
    Single-turn verified chat endpoint.
    
    Returns verified answer with source attribution.
    Does NOT allow guessing or unverified responses.
    
    Args:
        query: ChatRequest with user query
        llm_client: LLM client (for context, not primary answering)
        
    Returns:
        JSON with verified answer and sources
    """
    try:
        logger.info(f"Verify request: {query.text[:50]}...")
        
        # Get verification orchestrator
        orchestrator = get_verification_orchestrator()
        
        # Verify query
        verified_answer = await orchestrator.verify_and_answer(query.text)
        
        # Return structured JSON response
        result = StructuredOutputFormatter.format_structured_json(verified_answer)
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Verification failed",
                "message": str(e),
                "answer": "Unable to verify this query. Please try again with different wording.",
            }
        )


@router.websocket("/chat/verify/stream")
async def chat_verify_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming verified responses.
    
    ATTENTION: Applies mandatory web verification to all queries.
    - Searches multiple sources
    - Cross-validates facts
    - For locations: validates hierarchy via OpenStreetMap
    - Never returns unverified guesses
    
    Message format:
    {
        "text": "User query",
        "mode": "verify" // forces verification
    }
    """
    try:
        await websocket.accept()
        logger.info("✓ Verified WebSocket connection accepted")
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Verify request: {data.get('text', '')[:50]}...")
                
                # Get dependencies
                from backend.api.deps import get_llm_client, get_chat_history
                
                llm_gen = get_llm_client()
                llm_client = next(llm_gen)
                
                chat_gen = get_chat_history()
                chat_history = next(chat_gen)
                
                # Stream verified response
                await stream_verified_response(
                    websocket,
                    llm_client,
                    ChatRequest(**data),
                    chat_history
                )
                    
            except Exception as e:
                logger.error(f"Error in verified streaming: {e}", exc_info=True)
                await websocket.send_json({
                    "error": str(e),
                    "message": "Verification failed. Please try again."
                })
                break
                
    except WebSocketDisconnect:
        logger.info("Verified WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Verified WebSocket connection error: {e}", exc_info=True)


@router.websocket("/chat/verify/hybrid")
async def chat_verify_hybrid(websocket: WebSocket):
    """
    WebSocket endpoint for RAG + Web Verification hybrid approach.
    
    Strategy:
    1. Check internal RAG knowledge first
    2. Verify findings with web search
    3. Return combined verified answer
    
    This approach balances internal knowledge with web verification.
    """
    try:
        await websocket.accept()
        logger.info("✓ Hybrid RAG+Verified WebSocket connection accepted")
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Hybrid verify request: {data.get('text', '')[:50]}...")
                
                # Get dependencies
                from backend.api.deps import get_llm_client, get_chat_history, get_index
                
                llm_gen = get_llm_client()
                llm_client = next(llm_gen)
                
                chat_gen = get_chat_history()
                chat_history = next(chat_gen)
                
                index_gen = get_index()
                index = next(index_gen)
                
                # Stream hybrid verified response
                await stream_hybrid_rag_verified_response(
                    websocket,
                    llm_client,
                    ChatRequest(**data),
                    chat_history,
                    index
                )
                    
            except Exception as e:
                logger.error(f"Error in hybrid streaming: {e}", exc_info=True)
                await websocket.send_json({
                    "error": str(e),
                    "message": "Hybrid verification failed. Please try again."
                })
                break
                
    except WebSocketDisconnect:
        logger.info("Hybrid verified WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Hybrid verified WebSocket connection error: {e}", exc_info=True)


@router.get("/chat/verify/logs")
async def get_verification_logs(limit: int = Query(10, ge=1, le=100)):
    """
    Get verification logs for debugging.
    
    Args:
        limit: Number of logs to return (max 100)
        
    Returns:
        List of verification logs
    """
    from backend.verification.verification_logger import get_verification_logger
    
    logger_instance = get_verification_logger()
    recent_logs = logger_instance.get_recent_logs(limit)
    
    return JSONResponse({
        "total_logs": len(logger_instance.logs),
        "returned_logs": len(recent_logs),
        "logs": [
            {
                "query": log.query,
                "type": log.query_type,
                "timestamp": log.timestamp.isoformat(),
                "sources_count": log.search_results_count,
                "confidence": log.final_confidence,
                "answer_preview": log.final_answer[:100],
            }
            for log in recent_logs
        ],
        "statistics": logger_instance.get_statistics(),
    })


@router.get("/chat/verify/statistics")
async def get_verification_statistics():
    """
    Get verification statistics.
    
    Returns statistics about:
    - Total verified queries
    - Average confidence scores
    - Query type distribution
    - High confidence rate
    """
    from backend.verification.verification_logger import get_verification_logger
    
    logger_instance = get_verification_logger()
    stats = logger_instance.get_statistics()
    
    if not stats:
        return JSONResponse({
            "message": "No verification logs yet",
            "stats": {}
        })
    
    return JSONResponse(stats)

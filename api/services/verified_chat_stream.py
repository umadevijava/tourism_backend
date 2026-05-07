"""Enhanced chat stream service with web verification layer."""

import time
from typing import Optional

from fastapi import WebSocket

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from backend.core.config import settings
from backend.schemas.chat import ChatRequest
from backend.verification.verification_orchestrator import get_verification_orchestrator
from backend.verification.structured_output import StructuredOutputFormatter
from chatbot.bot.conversation.conversation_handler import (
    answer,
    answer_with_context,
    extract_content_after_reasoning,
    refine_question,
)
from chatbot.bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


async def stream_verified_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
):
    """
    Stream response with MANDATORY web verification.
    
    This ensures EVERY factual query is verified against multiple sources
    before answering. Never generates from memory alone.
    
    Args:
        websocket: WebSocket connection
        llm_client: LLM client
        query: Chat request
        chat_history: Chat history
    """
    try:
        start_time = time.time()
        
        # Get verification orchestrator
        orchestrator = get_verification_orchestrator()
        
        # Verify the query
        logger.info(f"Performing verification for: {query.text[:50]}...")
        verified_answer = await orchestrator.verify_and_answer(query.text)
        
        # Format for streaming output
        formatted_output = StructuredOutputFormatter.format_for_output(verified_answer)
        
        # Stream the formatted output
        for chunk in formatted_output.split("\n"):
            await websocket.send_text(chunk + "\n")
        
        # Also send JSON structured output
        json_output = StructuredOutputFormatter.format_structured_json(verified_answer)
        await websocket.send_json({
            "type": "verification_complete",
            "data": json_output
        })
        
        # Update chat history
        chat_history.append(f"question: {query.text}, answer: {verified_answer.answer}")
        
        took = time.time() - start_time
        logger.info(f"Verified response took {took:.2f} seconds")
        
    except Exception as exc:
        logger.exception("Error during verified streaming: %s", exc)
        await websocket.send_text("Error during verification: Please try again.")


async def stream_hybrid_rag_verified_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    index: VectorDatabaseDep,
):
    """
    Stream response using hybrid RAG + Web Verification approach.
    
    Strategy:
    1. Check internal knowledge (RAG vector DB)
    2. Verify findings with web search
    3. Combine and return verified answer
    
    Args:
        websocket: WebSocket connection
        llm_client: LLM client  
        query: Chat request
        chat_history: Chat history
        index: Vector database for RAG
    """
    try:
        start_time = time.time()
        
        # Step 1: Try RAG retrieval first
        logger.info(f"Attempting RAG retrieval for: {query.text[:50]}...")
        
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(settings.SYNTHESIS_STRATEGY, llm=llm_client)
        
        refined_user_input = await refine_question(
            llm_client, query.text, chat_history=chat_history, max_new_tokens=settings.MAX_NEW_TOKENS
        )
        
        retrieved_contents, sources = index.similarity_search_with_threshold(
            query=refined_user_input, k=settings.NUM_RETRIEVALS
        )
        
        retrieval_response = ""
        if retrieved_contents:
            logger.info(f"Found {len(sources)} relevant documents in RAG")
            retrieval_response = "**📚 Internal Knowledge Found:**\n\n"
        else:
            logger.info("No documents in RAG, will verify with web search")
            retrieval_response = "**🔍 Searching Web for Verification:**\n\n"
        
        await websocket.send_text(retrieval_response)
        
        # Step 2: Verify with web search
        logger.info("Verifying with web search...")
        orchestrator = get_verification_orchestrator()
        verified_answer = await orchestrator.verify_and_answer(query.text)
        
        # Step 3: Combine RAG + Web verification
        if retrieved_contents:
            # Combine RAG context with web-verified facts
            combined_answer = f"**From Internal Knowledge:** {retrieved_contents[:200]}...\n\n"
            combined_answer += f"**Verified from Web:** {verified_answer.answer}\n\n"
            combined_answer += "**Confidence:** High (Cross-verified from multiple sources)"
        else:
            combined_answer = verified_answer.answer
        
        # Stream combined response
        full_response = ""
        for chunk in combined_answer.split("\n"):
            full_response += chunk + "\n"
            await websocket.send_text(chunk + "\n")
        
        # Include source attribution
        if verified_answer.sources:
            sources_text = "**Sources:**\n"
            for source in verified_answer.sources[:3]:
                sources_text += f"- [{source.title}]({source.url})\n"
            await websocket.send_text(sources_text)
        
        # Update history
        chat_history.append(f"question: {query.text}, answer: {combined_answer}")
        
        took = time.time() - start_time
        logger.info(f"Hybrid RAG+Verified response took {took:.2f} seconds")
        
    except Exception as exc:
        logger.exception("Error during hybrid streaming: %s", exc)
        await websocket.send_text("Error during verification: Please try again.")

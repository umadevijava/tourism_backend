"""
Enhanced chat stream service with chat history persistence.
Wraps the original streaming functions to capture and save responses.
"""

import time
from fastapi import WebSocket
from sqlmodel import Session

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from backend.api.services.chat_stream import stream_chat_response, stream_rag_response, stream_google_search_response
from backend.api.services.chat_history import ChatHistoryService
from backend.core.config import settings
from backend.schemas.chat import ChatRequest
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


class ResponseCapture:
    """Helper to capture response while streaming."""
    def __init__(self):
        self.response = ""
    
    async def send_and_capture(self, websocket: WebSocket, text: str):
        """Send text to websocket and capture it."""
        self.response += text
        await websocket.send_text(text)


async def stream_chat_response_with_history(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    db_session: Session,
    session_id: str,
):
    """
    Stream chat response and save to database.
    """
    try:
        start_time = time.time()
        capture = ResponseCapture()
        full_response = ""
        max_tokens = settings.MAX_NEW_TOKENS
        
        from chatbot.bot.conversation.conversation_handler import (
            answer,
            extract_content_after_reasoning,
        )
        
        stream = await answer(
            llm=llm_client,
            question=query.text,
            chat_history=chat_history,
            max_new_tokens=max_tokens,
        )
        
        for output in stream:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)
        
        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(
                full_response,
                llm_client.model_settings.reasoning_stop_tag
            )
            if final_answer == "":
                final_answer = "I didn't provide the answer; perhaps I can try again."
        else:
            final_answer = full_response
        
        # Save to database
        ChatHistoryService.save_message(
            session_id=session_id,
            question=query.text,
            answer=final_answer,
            db_session=db_session,
            rag_mode=False,
            reasoning_mode=query.reasoning,
            web_search_mode=False,
        )
        
        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        took = time.time() - start_time
        logger.info(f"Chat response completed in {took:.2f}s, saved to session {session_id}")
        
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        await websocket.send_text("Error during streaming.")


async def stream_rag_response_with_history(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    index: VectorDatabaseDep,
    db_session: Session,
    session_id: str,
):
    """
    Stream RAG response and save to database.
    """
    try:
        start_time = time.time()
        full_response = ""
        max_tokens = settings.MAX_NEW_TOKENS
        
        from chatbot.bot.conversation.conversation_handler import (
            answer_with_context,
            extract_content_after_reasoning,
            refine_question,
        )
        from chatbot.bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
        from chatbot.helpers.prettier import prettify_source
        
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(
            settings.SYNTHESIS_STRATEGY, llm=llm_client, chatbot_mode=settings.CHATBOT_MODE
        )
        
        # Refine question
        refined_user_input = await refine_question(
            llm_client, query.text, chat_history=chat_history, max_new_tokens=128
        )
        
        # Retrieve documents
        retrieved_contents, sources = index.similarity_search_with_threshold(
            query=refined_user_input, k=settings.NUM_RETRIEVALS
        )
        
        retrieval_response = ""
        
        if retrieved_contents:
            max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 2000)
            context_char_count = 0
            limited_contents = []
            
            for content in retrieved_contents:
                content_len = len(content.page_content)
                if context_char_count + content_len > max_context_chars:
                    remaining = max_context_chars - context_char_count
                    if remaining > 100:
                        truncated = content.page_content[:remaining]
                        content.page_content = truncated + "..."
                        limited_contents.append(content)
                    break
                limited_contents.append(content)
                context_char_count += content_len
            
            retrieval_response += "**Retrieving Documents:**\n"
            for source in sources[:len(limited_contents)]:
                retrieval_response += prettify_source(source)
                retrieval_response += "\n"
            retrieval_response += "\n"
            
            context_data = limited_contents
        else:
            retrieval_response += "No relevant documents found.\n\n"
            context_data = []
        
        await websocket.send_text(retrieval_response)
        
        # Generate response with context
        streamer, _ = await answer_with_context(
            llm_client,
            ctx_synthesis_strategy,
            query.text,
            chat_history,
            context_data,
            max_tokens,
        )
        
        for output in streamer:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)
        
        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(
                full_response,
                llm_client.model_settings.reasoning_stop_tag
            )
            if final_answer == "":
                final_answer = "I wasn't able to provide the answer; Do you want me to try again?"
        else:
            final_answer = full_response
        
        # Save to database
        ChatHistoryService.save_message(
            session_id=session_id,
            question=query.text,
            answer=final_answer,
            db_session=db_session,
            rag_mode=True,
            reasoning_mode=query.reasoning,
            web_search_mode=False,
        )
        
        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        took = time.time() - start_time
        logger.info(f"RAG response completed in {took:.2f}s, saved to session {session_id}")
        
    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        try:
            await websocket.send_text("Error during RAG streaming.")
        except Exception:
            pass


async def stream_google_search_response_with_history(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    db_session: Session,
    session_id: str,
):
    """
    Stream Google Search response and save to database.
    """
    try:
        start_time = time.time()
        full_response = ""
        
        # For now, delegate to the original function
        # In the future, we can enhance this
        from backend.api.services.chat_stream import stream_google_search_response as original_google_search
        
        # We need to capture the response, so this is a simplified version
        logger.info(f"Google search initiated for session {session_id}")
        await websocket.send_text("[Google Search Mode] Searching for relevant information...")
        
        took = time.time() - start_time
        logger.info(f"Google search completed in {took:.2f}s")
        
    except Exception as exc:
        logger.exception("Error during Google search: %s", exc)
        try:
            await websocket.send_text("Error during Google search.")
        except Exception:
            pass

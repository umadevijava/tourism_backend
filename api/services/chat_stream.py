import time

from fastapi import WebSocket

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from backend.core.config import settings
from backend.schemas.chat import ChatRequest
from chatbot.bot.conversation.conversation_handler import (
    answer,
    answer_with_context,
    extract_content_after_reasoning,
    refine_question,
)
from chatbot.bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
from chatbot.bot.client.tool_calling import ToolCallingHandler
from chatbot.helpers.log import get_logger
from chatbot.helpers.prettier import prettify_source

logger = get_logger(__name__)


# TODO: https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567672
async def stream_chat_response(
    websocket: WebSocket, llm_client: LamaCppClientDep, query: ChatRequest, chat_history: ChatHistoryDep
):
    """
    Helper function to stream chat responses token by token.
    
    Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistoryDep): The chat history dependency to maintain conversation context.
    """
    try:
        start_time = time.time()

        full_response = ""
        has_sent_tokens = False
        # Use full token limit for accurate responses
        max_tokens = settings.MAX_NEW_TOKENS
        
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
                has_sent_tokens = True

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I didn't provide the answer; perhaps I can try again."
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        logger.debug(f"Updated chat history: {chat_history}")

        took = time.time() - start_time
        logger.info(f"\n--- Chat Response Took {took:.2f} seconds ---")
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        # Only send error message if we haven't sent any tokens yet
        # If tokens were already sent, closing the connection cleanly is enough
        if not has_sent_tokens:
            try:
                await websocket.send_json({"error": "Error during streaming."})
            except Exception:
                pass  # Connection may be closed


# TODO: https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567672
async def stream_rag_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    index: VectorDatabaseDep,
):
    """
    Helper function to stream RAG responses token by token.
    Optimized for speed with context limiting.
    
    Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LlamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistoryDep): The chat history dependency to maintain conversation context.
        index (VectorDatabaseDep): The vector database dependency for retrieval.
    """
    try:
        start_time = time.time()
        has_sent_tokens = False
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(
            settings.SYNTHESIS_STRATEGY, llm=llm_client, chatbot_mode=settings.CHATBOT_MODE
        )

        retrieval_response = ""
        full_response = ""
        max_tokens = settings.MAX_NEW_TOKENS

        # Refine question for better document retrieval accuracy
        refined_user_input = await refine_question(
            llm_client, query.text, chat_history=chat_history, max_new_tokens=128  # Reduced for speed
        )
        
        # Retrieve documents - NUM_RETRIEVALS is now optimized
        retrieved_contents, sources = index.similarity_search_with_threshold(
            query=refined_user_input, k=settings.NUM_RETRIEVALS
        )
        
        if retrieved_contents:
            # Limit context to MAX_CONTEXT_CHARS for faster LLM processing
            max_context_chars = getattr(settings, 'MAX_CONTEXT_CHARS', 2000)
            context_char_count = 0
            limited_contents = []
            
            for content in retrieved_contents:
                content_len = len(content.page_content)
                if context_char_count + content_len > max_context_chars:
                    # Truncate to fit within limit
                    remaining = max_context_chars - context_char_count
                    if remaining > 100:  # Only include if at least 100 chars available
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
        has_sent_tokens = True

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
                has_sent_tokens = True

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I wasn't able to provide the answer; Do you want me to try again?"
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")

        took = time.time() - start_time
        logger.info(f"\n--- RAG Response Took {took:.2f} seconds ---")

    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        # Only send error message if we haven't sent any tokens yet
        if not has_sent_tokens:
            try:
                await websocket.send_json({"error": "Error during RAG streaming."})
            except Exception:
                pass  # Connection may be closed
        except Exception:
            pass  # Connection may be closed


async def stream_google_search_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
):
    """
    Stream response with ACCURATE automatic Google Search tool calling.
    
    Optimized for accuracy:
    - Searches for factual and current information
    - Gets multiple results for comprehensive answers
    - Uses full token limits for complete responses
    - Includes proper source attribution
    
    Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistoryDep): The chat history dependency to maintain conversation context.
    """
    try:
        start_time = time.time()
        
        # Initialize tool calling handler
        tool_handler = ToolCallingHandler(llm_client)
        
        prompt = query.text
        tool_results = []
        full_response = ""
        
        # Check if tools are needed (includes factual queries for accuracy)
        if tool_handler.detect_tool_need(prompt):
            logger.info(f"Executing web search for accurate information: {prompt[:50]}...")
            
            # Extract search terms
            search_terms = tool_handler._extract_search_terms(prompt)
            if search_terms:
                search_term = search_terms[0]
                search_info = f"**🔍 Searching for:** {search_term}\n"
                await websocket.send_text(search_info)
                
                # Execute search (cached or fresh)
                from chatbot.bot.client.tool_calling import _SEARCH_CACHE
                cache_key = search_term.lower()
                
                if cache_key in _SEARCH_CACHE:
                    logger.info(f"Using cached results for: {search_term}")
                    result = _SEARCH_CACHE[cache_key]
                else:
                    logger.info(f"Fetching web results for: {search_term}")
                    # Get comprehensive results (8-10) for accuracy
                    result = await tool_handler.google_search.search(search_term, num_results=8)
                    _SEARCH_CACHE[cache_key] = result
                
                tool_results.append({
                    "tool": "google_search",
                    "input": {"query": search_term},
                    "output": result,
                })
                
                # Format search results comprehensively - ENSURE LLM USES THEM
                if result.get("success"):
                    results_list = result.get("results", [])
                    # Create PROMINENT search context that LLM will prioritize
                    tool_context = f"\n\n=== AUTHORITATIVE WEB SEARCH RESULTS FOR: '{search_term}' ==="
                    tool_context += f"\n(Found {len(results_list)} results - Use these facts to answer the user's question)\n\n"
                    
                    for i, r in enumerate(results_list[:8], 1):
                        title = r.get('title', 'No title')
                        snippet = r.get('snippet', 'No description')
                        url = r.get('url', 'No URL')
                        tool_context += f"Result {i}: {title}\n"
                        tool_context += f"Content: {snippet}\n"
                        tool_context += f"Source: {url}\n\n"
                    
                    tool_context += "=== AUTHORITATIVE WEB SEARCH RESULTS ===\n"
                    tool_context += "(For location/landmark/district questions, use ONLY the results below - they are current and accurate)\n\n"
                    # Put search results at START with explicit priority
                    prompt = f"{tool_context}\nUSER QUESTION: {prompt}\n\n[Instructions]: For factual location/landmark/district information, base your answer ONLY on the search results above. Answer directly and comprehensively:"
        
        # Stream response with full tokens for accuracy
        max_tokens = settings.MAX_NEW_TOKENS
        stream = await answer(
            llm=llm_client,
            question=prompt,
            chat_history=chat_history,
            max_new_tokens=max_tokens,
        )
        
        for output in stream:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)
        
        # Update history
        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I was unable to provide a complete answer."
        else:
            final_answer = full_response
        
        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        
        took = time.time() - start_time
        logger.info(f"\n--- Web Search Response (Accurate) Took {took:.2f} seconds ---")
        
    except Exception as exc:
        logger.exception("Error during web search streaming: %s", exc)
        await websocket.send_text("\nError during web search. Please try again.")
        
        # Log execution time
        took = time.time() - start_time
        logger.info(f"\n--- Google Search Response Took {took:.2f} seconds ---")
        if tools_used:
            logger.info(f"Tools used: {len(tools_used)}")
    
    except Exception as exc:
        logger.exception("Error during Google Search streaming: %s", exc)
        await websocket.send_text("Error during Google Search streaming.")

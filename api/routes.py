from fastapi import APIRouter

from backend.api.endpoints import chat, chat_stream, documents, health, agents, verified_chat, history

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="", tags=["chat"])
api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(chat_stream.router, prefix="", tags=["chat-stream"])
api_router.include_router(verified_chat.router, prefix="", tags=["verified-chat"])
api_router.include_router(agents.router, prefix="", tags=["agents"])
api_router.include_router(history.router, prefix="", tags=["history"])

"""
Global application state.
Holds singleton instances that are initialized during app startup.
"""


from sqlalchemy import Engine

from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.memory.vector_database.chroma import Chroma

# Global singleton instances
engine: Engine | None = None
llm_client: LamaCppClient | None = None
index: Chroma | None = None

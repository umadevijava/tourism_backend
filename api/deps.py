"""\nDefines dependencies used by the endpoints.
"""

from typing import Annotated, Generator

from fastapi import Depends
from sqlmodel import Session

from backend import state
from backend.chat_history import chat_history
from backend.core.config import settings
from backend.database import create_db_engine
from backend.llm_client import create_llm_client
from backend.vector_database import init_index
from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.conversation.chat_history import ChatHistory
from chatbot.bot.memory.vector_database.chroma import Chroma


def get_llm_client() -> Generator[LamaCppClient, None, None]:
    """
    Dependency to get the LLM client instance with lazy initialization.
    """
    if state.llm_client is None:
        state.llm_client = create_llm_client(settings.MODEL_FOLDER)
    yield state.llm_client


def get_chat_history() -> Generator[ChatHistory, None, None]:
    """
    Dependency to get the chat history instance.
    """
    yield chat_history


def get_index() -> Generator[Chroma, None, None]:
    """
    Dependency to get the vector database index instance with lazy initialization.
    """
    if state.index is None:
        state.index = init_index(settings.VECTOR_STORE_PATH)
    yield state.index


def get_db_session() -> Generator[Session, None, None]:
    """
    Create a new database session and close the session after the operation has ended.
    """
    if state.engine is None:
        state.engine = create_db_engine()
    with Session(state.engine) as session:
        yield session


LamaCppClientDep = Annotated[LamaCppClient, Depends(get_llm_client)]
ChatHistoryDep = Annotated[ChatHistory, Depends(get_chat_history)]
VectorDatabaseDep = Annotated[Chroma, Depends(get_index)]
SessionDep = Annotated[Session, Depends(get_db_session)]

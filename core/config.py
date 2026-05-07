from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_PATH / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Chatbot API"
    VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    MODEL_FOLDER: Path = ROOT_PATH / "models"
    VECTOR_STORE_PATH: Path = ROOT_PATH / "vector_store" / "docs_index"
    DOCS_PATH: Path = ROOT_PATH / Path("docs")

    DATABASE_URL: str = f"sqlite:///{ROOT_PATH / 'vector_store' / 'registry.db'}"

    # LLM Model Configuration
    MODEL: str = "llama-3.2:1b"
    MAX_NEW_TOKENS: int = 512

    # Retrieval Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SYNTHESIS_STRATEGY: str = "tree-summarization"
    NUM_RETRIEVALS: int = 3  # Reduced from 2 for faster retrieval
    CHUNK_SIZE: int = 500  # Reduced from 1000 for faster processing
    CHUNK_OVERLAP: int = 50
    
    # Response optimization
    MAX_CONTEXT_CHARS: int = 2000  # Limit context size to 2000 chars for faster LLM processing
    RESPONSE_TIMEOUT_SECONDS: int = 60  # Timeout for long-running queries

    # Chat History Configuration
    CHAT_HISTORY_LENGTH: int = 2
    
    # Chatbot Mode Configuration (e.g., "tourism", "general")
    CHATBOT_MODE: str = "general"

    # WebSocket Configuration
    WEBSOCKET_MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB

    # File Upload Configuration
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = [".md", ".pdf", ".docx", ".txt", ".html", ".doc"]


settings = Settings()

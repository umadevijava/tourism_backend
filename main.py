from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import state
from backend.api.routes import api_router
from backend.core.config import settings
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Defer all initialization until first use
    state.engine = None
    state.llm_client = None
    state.index = None

    yield

    # Cleanup
    if state.engine:
        try:
            state.engine.dispose()
            logger.info("Database engine disposed")
        except Exception as e:
            logger.warning(f"Error disposing database engine: {e}")
    if state.llm_client:
        try:
            state.llm_client.close()
            logger.info("LLM client closed")
        except Exception as e:
            logger.warning(f"Error closing LLM client: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint - returns API info"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

# Note: A single Uvicorn worker is probably what you would want to use when using a distributed container
# management system like Kubernetes.

if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host=settings.HOST,
        port=settings.PORT,
        # log_config=None,
        # workers=max(1, os.cpu_count() - 1),
        workers=1,
    )

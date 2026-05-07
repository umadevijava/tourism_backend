from fastapi import APIRouter

from backend.core.config import settings
from backend.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(version=settings.VERSION)

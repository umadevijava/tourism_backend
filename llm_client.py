from pathlib import Path

from backend.core.config import settings
from chatbot.bot.client.lama_cpp_client import LamaCppClient
from chatbot.bot.model.model_registry import get_model_settings


def create_llm_client(model_folder: Path) -> LamaCppClient:
    settings.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    model_settings = get_model_settings(settings.MODEL)

    return LamaCppClient(model_folder=model_folder, model_settings=model_settings)

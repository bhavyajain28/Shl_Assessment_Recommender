"""Central configuration, loaded from environment variables / .env."""
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

_PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider: "groq" (default, free tier), "openrouter", or "openai".
    # Any OpenAI-compatible endpoint works by overriding llm_base_url / llm_model.
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 18.0
    llm_max_retries: int = 2

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    catalog_json_path: Path = BASE_DIR / "data" / "catalog.json"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"

    max_conversation_turns: int = 8
    max_recommendations: int = 10
    min_recommendations: int = 1
    top_k_retrieval: int = 25

    log_level: str = "INFO"

    @property
    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        return _PROVIDER_DEFAULTS.get(self.llm_provider, _PROVIDER_DEFAULTS["groq"])["base_url"]

    @property
    def resolved_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return _PROVIDER_DEFAULTS.get(self.llm_provider, _PROVIDER_DEFAULTS["groq"])["model"]


@lru_cache
def get_settings() -> Settings:
    return Settings()

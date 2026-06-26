from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    app_name: str = "Family AI Assistant"
    app_version: str = "0.1.0"

    database_url: str = "postgresql://postgres:postgres@localhost:5433/family_ai_assistant"
    database_echo: bool = False

    cors_origins: list[str] = ["*"]

    # ── Telegram bot integration ──
    # All required for the bot to function; the service starts fine without
    # them (telegram endpoints return 503) so dev environments don't have to
    # set them.
    telegram_bot_token: str = ""

    # ── OpenAI (intent extraction) ──
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Stronger model for the "family brain" free-form Q&A over the full family
    # snapshot (more reasoning headroom than the cheap intent-extraction model).
    openai_brain_model: str = "gpt-4o"

    # ── family-os REST API (the bot calls this to create events/grocery) ──
    family_os_api_url: str = "https://family-os-4ilvxexrha-zf.a.run.app"
    family_os_service_token: str = ""

@lru_cache
def get_settings() -> Settings:
    return Settings()


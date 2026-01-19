import os
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

# Point to the root directory: arjundangle-vexo-rakwireless-chatbot/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Use SettingsConfigDict to ignore extra environment variables
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore", 
        case_sensitive=False
    )

    PROJECT_NAME: str = "RAK Knowledge Engine"
    VERSION: str = "1.0.0"
    
    # AI Config
    OPENAI_API_KEY: str
    # Removed OPENROUTER_BASE_URL as we are now using the default OpenAI endpoint
    MODEL_NAME: str = "gpt-4.1-nano-2025-04-14"
    
    # OPENROUTER_API_KEY: str
    # OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # MODEL_NAME: str = "mistralai/devstral-2512:free"
    
    # Vector DB Config
    DB_PATH: str = str(BASE_DIR / "storage" / "vector_db")
    COLLECTION_NAME: str = "rak_knowledge"

    # FIX: Define ALLOWED_ORIGINS to resolve the validation error
    ALLOWED_ORIGINS: List[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Converts comma-separated string from .env into a list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        return v

settings = Settings()
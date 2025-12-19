import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAK Knowledge Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # AI Config
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    MODEL_NAME: str = "mistralai/devstral-2512:free"
    
    # Vector DB Config
    # We navigate up from app/core/ to storage/
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH: str = os.path.join(BASE_DIR, "storage", "vector_db")
    COLLECTION_NAME: str = "rak_knowledge"

settings = Settings()
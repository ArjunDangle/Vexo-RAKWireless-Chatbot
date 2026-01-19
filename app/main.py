import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from app.services.chat_engine import ChatEngine
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAKwireless Knowledge Engine API")

# Enable CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Chat Engine
bot = None

@app.on_event("startup")
async def startup_event():
    global bot
    try:
        bot = ChatEngine()
        logger.info("✅ Chat Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Chat Engine: {e}")

# --- SCHEMAS ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

# --- ROUTES ---

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": bot is not None}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streaming Chat Endpoint.
    Instead of waiting 100s, this sends chunks of text as they are ready.
    """
    if not bot:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")

    try:
        # We return a StreamingResponse which executes the generator in chat_engine
        return StreamingResponse(
            bot.get_response_stream(request.message, request.history),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/static")
async def chat_static_endpoint(request: ChatRequest):
    """
    Standard non-streaming endpoint if you prefer traditional JSON.
    """
    if not bot:
        raise HTTPException(status_code=500, detail="Chat engine not initialized")
    
    try:
        response = await bot.get_response(request.message, request.history)
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
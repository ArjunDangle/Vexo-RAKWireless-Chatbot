import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our refined schemas
from app.models.chat import ChatRequest, ChatResponse
# Import our unified "Brain"
from app.services.chat_engine import ChatEngine
from app.core.config import settings

# --- INITIALIZATION ---
app = FastAPI(title="RAKwireless Knowledge Engine", version="1.0.0")

# Setup Logging for production visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Unified Engine once on startup to save latency
try:
    bot = ChatEngine()
    logger.info("✅ Unified RAK ChatEngine initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize ChatEngine: {e}")
    bot = None

# --- ROUTES ---

@app.get("/")
async def health_check():
    """Returns the status of the API and the engine."""
    return {
        "status": "online", 
        "engine": "RAK Unified v1.0",
        "ready": bot is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Unified endpoint for processing technical queries.
    Uses Query Rewriting, HyDE, Hybrid Search, and Reranking.
    """
    if not bot:
        logger.error("Attempted to call /chat but engine is not initialized.")
        raise HTTPException(status_code=500, detail="Chat Engine not initialized.")
    
    try:
        logger.info(f"Incoming query: {request.message}")
        
        # Process the query using the unified service
        # History is passed for contextual query rewriting
        result = bot.get_response(request.message, history=request.history)
        
        return ChatResponse(
            answer=result['answer'],
            sources=result['sources'],
            latency=result['latency']
        )
        
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Host on 0.0.0.0 to allow containerization or external access
    uvicorn.run(app, host="0.0.0.0", port=8000)
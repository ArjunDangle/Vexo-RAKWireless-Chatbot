from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.chat import ChatRequest, ChatResponse
from chat import RAKChatEngine  # Ensure chat.py is in your root or PYTHONPATH
import logging

# --- INITIALIZATION ---
app = FastAPI(title="RAKwireless Knowledge Engine")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for your frontend (React/Vue/etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Elite Engine once on startup
try:
    bot = RAKChatEngine()
    logger.info("✅ Elite RAKChatEngine initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAKChatEngine: {e}")
    bot = None

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "online", "engine": "RAK Elite v1.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not bot:
        raise HTTPException(status_code=500, detail="Chat Engine not initialized.")
    
    try:
        logger.info(f"Processing query: {request.message}")
        
        # This calls the method we built in Phase 3, 4, and 5
        # It handles HyDE, Context Swapping, and Citations internally
        answer_with_citations = bot.ask(request.message)
        
        # Extract sources from the engine logic to return as a clean list
        _, sources = bot.retrieve_context(request.message)
        
        return ChatResponse(
            answer=answer_with_citations,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_engine import ChatEngine

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine (Global Singleton)
chat_engine = ChatEngine()

@app.get("/")
def root():
    return {"message": "RAK Knowledge Engine is Online 🚀"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_data = chat_engine.get_response(request.query)
        return ChatResponse(**response_data)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
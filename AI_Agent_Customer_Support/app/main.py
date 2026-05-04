from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.rag import RAGPipeline
from app.history import HistoryManager
from app.logger import logger

app = FastAPI(title="Customer Support Chatbot")

# Initialize components
rag = RAGPipeline()
history = HistoryManager()

# Hard-coded credentials
USERNAME = "user"
PASSWORD = "pass"

# Track logged-in users
logged_in_users = set()  # Set of usernames

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    username: str

class ChatResponse(BaseModel):
    answer: str

class LoginRequest(BaseModel):
    username: str
    password: str

class HistoryRequest(BaseModel):
    username: str

class HistoryResponse(BaseModel):
    history: List[Dict]

class Token(BaseModel):
    message: str

@app.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """Log in with username and password."""
    if request.username != USERNAME or request.password != PASSWORD:
        logger.warning("Invalid login")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logged_in_users.add(request.username)
    logger.info(f"Login successful: {request.username}")
    return {"message": "Login successful"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat queries for logged-in users."""
    if request.username not in logged_in_users:
        logger.warning(f"Not logged in: {request.username}")
        raise HTTPException(status_code=401, detail="Not logged in")
    
    documents = rag.hybrid_retrieval(request.query)
    context = rag.compress_context(documents)
    chat_history = history.get_history(request.username)
    answer = rag.generate_response(request.query, context, chat_history)
    
    history.add_message(request.username, {"role": "user", "content": request.query})
    history.add_message(request.username, {"role": "assistant", "content": answer})
    
    logger.info(f"Chat response for {request.username}")
    return ChatResponse(answer=answer)

@app.post("/history", response_model=HistoryResponse)
async def get_history(request: HistoryRequest):
    """Retrieve chat history for a logged-in user."""
    if request.username not in logged_in_users:
        logger.warning(f"Not logged in: {request.username}")
        raise HTTPException(status_code=401, detail="Not logged in")
    
    chat_history = history.get_history(request.username)
    logger.info(f"History retrieved for {request.username}")
    return HistoryResponse(history=chat_history)

@app.post("/logout", response_model=Token)
async def logout(request: HistoryRequest):
    """Log out and clear user session."""
    if request.username not in logged_in_users:
        logger.warning(f"Not logged in: {request.username}")
        raise HTTPException(status_code=401, detail="Not logged in")
    
    logged_in_users.remove(request.username)
    logger.info(f"Logout successful: {request.username}")
    return {"message": "Logout successful"}
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
import os
from models.model_loader import generate_response
from models.db_manager import MongoDBManager
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="VisiQ-GPT API",
    description="API for processing text, images, and PDFs using LLM models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryParams(BaseModel):
    model_name: Optional[str] = "llama3.2-vision"
    max_tokens: Optional[int] = 800
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    detailed_response: Optional[bool] = True

class TextQuery(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    model_name: str = "llama3.2-vision"
    max_tokens: int = 800
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    detailed_response: bool = True

class SessionManager:
    def __init__(self):
        self.db_manager = MongoDBManager()

    def create_session(self):
        return self.db_manager.create_session()

    def get_session_file(self, session_id):
        return self.db_manager.get_session_file(session_id)

    def update_session_file(self, session_id, file_path, file_type):
        return self.db_manager.update_session_file(session_id, file_path, file_type)

    def add_conversation(self, session_id, prompt, response):
        return self.db_manager.add_conversation_history(session_id, prompt, response)

session_manager = SessionManager()

@app.post("/api/session")
async def create_session():
    """Create a new chat session"""
    session_id = session_manager.create_session()
    return {"session_id": session_id}

# New endpoint for direct text queries
@app.post("/api/chat")
async def chat(query: TextQuery):
    """Endpoint for direct text queries without file upload"""
    try:
        response = generate_response(
            prompt=query.prompt,
            model_name=query.model_name,
            max_tokens=query.max_tokens,
            temperature=query.temperature,
            top_p=query.top_p,
            top_k=query.top_k,
            detailed_response=query.detailed_response
        )

        if query.session_id:
            session_manager.add_conversation(query.session_id, query.prompt, response)

        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

# Modified file upload endpoint
@app.post("/api/upload")
async def upload_and_query(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: Optional[str] = Form(None),
    model_name: str = Form("llama3.2-vision"),
    max_tokens: int = Form(800),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    detailed_response: bool = Form(True)
):
    """Endpoint for file upload and query"""
    temp_file_path = None
    try:
        params = {
            "model_name": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "detailed_response": detailed_response
        }

        # Save uploaded file
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Update session with new file if session exists
        if session_id:
            file_type = 'pdf' if file.filename.lower().endswith('.pdf') else 'image'
            session_manager.update_session_file(session_id, temp_file_path, file_type)

        # Process file
        if file.filename.lower().endswith('.pdf'):
            response = generate_response(
                prompt=prompt,
                pdf_path=temp_file_path,
                **params
            )
        else:  # Assume image
            response = generate_response(
                prompt=prompt,
                image_path=temp_file_path,
                **params
            )

        if session_id:
            session_manager.add_conversation(session_id, prompt, response)

        return {"status": "success", "response": response}

    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path) and not session_id:
            os.remove(temp_file_path)

# Modified session query endpoint
@app.post("/api/session/query")
async def session_query(query: TextQuery):
    """Endpoint for querying within an existing session"""
    try:
        if not query.session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        session_file = session_manager.get_session_file(query.session_id)
        if not session_file:
            return await chat(query)

        params = {
            "model_name": query.model_name,
            "max_tokens": query.max_tokens,
            "temperature": query.temperature,
            "top_p": query.top_p,
            "top_k": query.top_k,
            "detailed_response": query.detailed_response
        }

        if session_file['file_type'] == 'pdf':
            response = generate_response(
                prompt=query.prompt,
                pdf_path=session_file['file_path'],
                **params
            )
        else:
            response = generate_response(
                prompt=query.prompt,
                image_path=session_file['file_path'],
                **params
            )

        session_manager.add_conversation(query.session_id, query.prompt, response)
        return {"status": "success", "response": response}

    except Exception as e:
        logger.error(f"Error processing session query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

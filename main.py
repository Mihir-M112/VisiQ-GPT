# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from models.model_loader import generate_response
import shutil
import os

app = FastAPI()

# Endpoint for text-based query
@app.post("/generate-text/")
async def generate_text(
    prompt: str = Form(...),
    model_name: str = Form("llama3.2-vision"),
    max_tokens: int = Form(300),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    system_prompt: str = Form("Give answer in short.")  # Default value
):
    try:
        response = generate_response(prompt, model_name, None, max_tokens, temperature, top_p, top_k, system_prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint for image-based query
@app.post("/generate-image/")
async def generate_image(prompt: str = Form(...), file: UploadFile = File(...), model_name: str = Form("llama3.2-vision")):
    # Save the uploaded image temporarily
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call model with image
        response = generate_response(prompt, model_name, image_path=temp_file_path)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)




@app.get("/health/")
def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "running"}


# Root endpoint for quick check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Model API!"}

"""API routes for SPRUT 3.0."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.core.config import settings

router = APIRouter()

class ProcessRequest(BaseModel):
    message: dict
    user_id: Optional[int] = None

class ProcessResponse(BaseModel):
    text: str
    voice_data: Optional[str] = None
    chunks: Optional[list[str]] = None

@router.get("/health")
async def health():
    return {"status": "healthy", "service": "sprut-api", "version": "3.0.0"}

@router.post("/api/process", response_model=ProcessResponse)
async def process_message(
    request: ProcessRequest,
    authorization: str = Header(default=""),
):
    expected = f"Bearer {settings.api_secret_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if request.user_id and request.user_id not in settings.allowed_user_id_list:
        raise HTTPException(status_code=403, detail="User not authorized")
    # Placeholder - will be wired in Task 16
    return ProcessResponse(text="SPRUT 3.0 is running. Processing not yet implemented.")

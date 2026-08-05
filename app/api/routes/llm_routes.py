from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_service import LLMService

router = APIRouter()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/generate")
def generate_text(request: PromptRequest):
    response = LLMService.generate_response(request.prompt)
    return {
        "response": response
    }

import ollama
from app.core.config import settings

class LLMService:
    @staticmethod
    def generate_response(prompt: str):
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response["message"]["content"]

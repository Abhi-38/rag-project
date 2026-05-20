from fastapi import FastAPI

from app.core.config import settings
from app.api.routes.llm_routes import router as llm_router

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0"
)

app.include_router(
    llm_router,
    prefix="/api/v1/llm",
    tags=["LLM"]
)


@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.ENV
    }
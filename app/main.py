from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0"
)


@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.ENV
    }
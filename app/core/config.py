from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str
    ENV: str

    LLM_MODEL: str

    EMBEDDING_MODEL: str

    VECTOR_DB: str

    class Config:
        env_file = ".env"


settings = Settings()
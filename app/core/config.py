import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = os.getenv("APP_NAME", "MedicalRAGAssistant")
    ENV: str = os.getenv("ENV", "development")
    
    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Embeddings & Vector DB
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    VECTOR_DB: str = os.getenv("VECTOR_DB", "chromadb")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "./vectorstore")
    
    # Chunking & Retrieval
    PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "1000"))
    PARENT_CHUNK_OVERLAP: int = int(os.getenv("PARENT_CHUNK_OVERLAP", "150"))
    CHILD_CHUNK_SIZE: int = int(os.getenv("CHILD_CHUNK_SIZE", "250"))
    CHILD_CHUNK_OVERLAP: int = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
    
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "15"))
    TOP_K_RERANKED: int = int(os.getenv("TOP_K_RERANKED", "5"))

settings = Settings()

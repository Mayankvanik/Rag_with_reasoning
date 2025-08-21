import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "conversational_rag")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 4))
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 5))
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-3.5-turbo"

config = Config()
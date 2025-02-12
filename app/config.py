import os
from dotenv import load_dotenv

load_dotenv()

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://zero:26692669@localhost:5432/fastbot_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 1024))


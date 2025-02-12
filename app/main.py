import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# from app.config import REDIS_URL
from .database import async_engine, Base
# import redis.asyncio as redis

# Import routers
from .routers import chat, file_upload, scrape, metrics

logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis
    # app.state.redis_client = redis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Startup complete.")
    yield
    # await app.state.redis_client.close()
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan, title="File Chat & Knowledgebase API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(file_upload.router, prefix="/file", tags=["file"])
app.include_router(scrape.router, prefix="/url", tags=["scrape"])
app.include_router(metrics.router, prefix="/view", tags=["metrics"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4, reload=True)

# main.py
import io
import os
import uuid
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from dotenv import load_dotenv
from config import MODEL_PROVIDER
# from models.openai_model import get_embedding, get_title_and_summary
from models.embeddings import get_embedding
from models.gemini_model import get_title_and_summary



load_dotenv()
# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("file-chat-api")

# --------------------------
# OpenAI API Configuration
# --------------------------
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set

# --------------------------
# File Extraction & Related Imports
# --------------------------
import fitz             # PyMuPDF for PDF extraction
import docx             # python-docx for DOCX files
import pandas as pd     # For Excel extraction
from PIL import Image   # Image processing
import pytesseract      # OCR for images

# --------------------------
# Qdrant Client
# --------------------------
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# --------------------------
# SQLAlchemy (Async) for PostgreSQL
# --------------------------
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, DateTime, Text, Index

DATABASE_URL = "postgresql+asyncpg://zero:26692669@localhost:5432/fastbot_db"  # Adjust as needed
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# --------------------------
# Redis (Async) for Caching
# --------------------------
import redis.asyncio as redis
REDIS_URL = "redis://localhost:6379"
redis_client = redis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)

# --------------------------
# Database Models with Indexing (and partitioning comments)
# --------------------------
class ChatThread(Base):
    __tablename__ = "chat_threads"
    id = Column(String, primary_key=True, index=True)
    organization_id = Column(String, index=True)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_chatthread_org", "organization_id"),
    )

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String, index=True)  # "user" or "assistant"
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_chatmessage_thread", "thread_id"),
        Index("idx_chatmessage_timestamp", "timestamp"),
    )

# ---
# Example DDL for Partitioning (to be executed separately by a DBA or migration tool):
#
# CREATE TABLE chat_messages (
#   id TEXT PRIMARY KEY,
#   thread_id TEXT NOT NULL,
#   role TEXT NOT NULL,
#   message TEXT,
#   timestamp TIMESTAMP NOT NULL
# ) PARTITION BY RANGE (timestamp);
#
# CREATE TABLE chat_messages_2023 PARTITION OF chat_messages
#   FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
#
# ---

# --------------------------
# Global Variables & Qdrant Initialization
# --------------------------
# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
VECTOR_DIM = 1024
TARGET_DIM = 1024

KNOWLEDGEBASE_COLLECTION = "knowledgebase"

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_qdrant_collection():
    """Ensure that the Qdrant collection exists; if not, create it."""
    try:
        _ = qdrant_client.get_collection(collection_name=KNOWLEDGEBASE_COLLECTION)
        logger.info("Qdrant collection '%s' exists.", KNOWLEDGEBASE_COLLECTION)
    except Exception as e:
        logger.warning("Qdrant collection not found; creating one: %s", e)
        qdrant_client.recreate_collection(
            collection_name=KNOWLEDGEBASE_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        logger.info("Qdrant collection '%s' created.", KNOWLEDGEBASE_COLLECTION)

ensure_qdrant_collection()

# --------------------------
# FastAPI Initialization & Middleware
# --------------------------
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = redis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Application startup complete.")
    yield
    if redis_client:
        await redis_client.close()
    logger.info("Application shutdown complete.")

app = FastAPI(lifespan=lifespan, title="Production‑Grade File Chat & Knowledgebase API with OpenAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Utility Functions: File Extraction
# --------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        logger.error("Error extracting PDF: %s", e)
        raise

def extract_text_from_doc(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(para.text for para in document.paragraphs)
    except Exception as e:
        logger.error("Error extracting DOCX: %s", e)
        raise

def extract_text_from_excel(file_bytes: bytes) -> str:
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    except Exception as e:
        logger.error("Error extracting Excel: %s", e)
        raise

def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error("Error performing OCR on image: %s", e)
        raise

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")  # Assuming UTF-8 encoding
    except Exception as e:
        logger.error("Error extracting TXT: %s", e)
        raise

def extract_text(file: UploadFile) -> str:
    ext = file.filename.split(".")[-1].lower()
    file_bytes = file.file.read()
    file.file.seek(0)

    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ["doc", "docx"]:
        return extract_text_from_doc(file_bytes)
    elif ext in ["xls", "xlsx"]:
        return extract_text_from_excel(file_bytes)
    elif ext in ["jpg", "jpeg", "png", "bmp"]:
        return extract_text_from_image(file_bytes)
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        logger.error("Unsupported file type: %s", ext)
        raise HTTPException(status_code=400, detail="Unsupported file type")
# --------------------------
# Utility Functions: Text Chunking & Asynchronous Title/Summary and Embedding
# --------------------------
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)
    return chunks

async def async_generate_embedding(text: str) -> List[float]:
    """
    Asynchronously split the text into chunks, generate embeddings for each chunk concurrently,
    and return the average embedding.
    """
    chunks = chunk_text(text)
    if not chunks:
        return [0.0] * VECTOR_DIM
    embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))
    # Average each dimension over all chunks
    avg_embedding = [float(sum(dim)) / len(dim) for dim in zip(*embeddings)]
    return avg_embedding

async def async_process_text(text: str, filename: str) -> (List[float], Dict[str, str]):
    """
    Run asynchronous tasks to generate an average embedding and obtain title/summary.
    For title/summary extraction, we send the first chunk.
    """
    embedding = await async_generate_embedding(text)
    chunks = chunk_text(text)
    title_summary = await get_title_and_summary(chunks[0] if chunks else text, filename)
    return embedding, title_summary

# --------------------------
# Utility Functions: Qdrant Ingestion
# --------------------------
def add_embedding_to_qdrant(organization_id: str, doc_id: str, embedding: List[float], metadata: dict):
    try:
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"organization_id": organization_id, **metadata}
        )
        qdrant_client.upsert(
            collection_name=KNOWLEDGEBASE_COLLECTION,
            points=[point]
        )
        logger.info("Upserted document %s into Qdrant for organization %s.", doc_id, organization_id)
    except Exception as e:
        logger.error("Error upserting into Qdrant: %s", e)
        raise

# --------------------------
# Synchronous Processing Function (used by Celery)
# --------------------------
def process_file_and_ingest_sync(organization_id: str, file: UploadFile) -> str:
    """
    Synchronous function (used in Celery tasks) to:
      - Extract text from the file
      - Run asynchronous processing to get embedding and title/summary
      - Upsert the document into Qdrant
    """
    text = extract_text(file)
    try:
        embedding, title_summary = asyncio.run(async_process_text(text, file.filename))
    except Exception as e:
        logger.error("Async processing error: %s", e)
        raise
    doc_id = str(uuid.uuid4())
    metadata = {
        "filename": file.filename,
        "text_snippet": text[:200]
    }
    metadata.update(title_summary)
    add_embedding_to_qdrant(organization_id, doc_id, embedding, metadata)
    return doc_id

# --------------------------
# Utility Functions: OpenAI Chat Completion for Replies
# --------------------------

# Depending on the provider specified in the config, import the appropriate module.
if MODEL_PROVIDER.lower() == "openai":
    from models.openai_model import generate_reply as generate_reply_openai
elif MODEL_PROVIDER.lower() == "gemini":
    from models.gemini_model import generate_reply as generate_reply_gemini
elif MODEL_PROVIDER.lower() == "deepseek":
    from models.deepseek_model import generate_reply as generate_reply_deepseek
else:
    raise ValueError("Invalid MODEL_PROVIDER specified in the .env file.")

async def generate_chat_reply(history, user_message):
    """
    Unified function to generate a reply based on conversation history and the latest user message.
    For OpenAI, we expect history to be a list of messages.
    For Gemini, we convert the history to a prompt string.
    """
    if MODEL_PROVIDER.lower() == "openai":
        # Convert history keys to match what OpenAI expects
        messages = []
        for entry in history:
            # If 'content' isn't present, fall back to 'message'
            content = entry.get("content", entry.get("message", ""))
            messages.append({"role": entry["role"], "content": content})
        messages.append({"role": "user", "content": user_message})
        return await generate_reply_openai(messages)
    
    elif MODEL_PROVIDER.lower() == "gemini":
        # For Gemini, convert the history (list of dicts) into a single prompt.
        # Check for both 'content' and 'message'
        prompt = "Conversation so far:\n"
        for entry in history:
            content = entry.get("content", entry.get("message", "[No content]"))
            prompt += f"{entry['role'].capitalize()}: {content}\n"
        prompt += f"User: {user_message}\nAssistant:"
        return generate_reply_gemini(prompt)
    
    elif MODEL_PROVIDER.lower() == "deepseek":
        # Prepare messages for Deepseek
        messages = []
        for entry in history:
            content = entry.get("content", entry.get("message", ""))
            messages.append({"role": entry["role"], "content": content})
        messages.append({"role": "user", "content": user_message})
        return await asyncio.to_thread(generate_reply_deepseek, messages)

    else:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


# --------------------------
# Database Dependency Helpers
# --------------------------
async def get_db_session() -> AsyncSession:
    async with async_session() as session:
        yield session

# --------------------------
# Pydantic Models for Chat
# --------------------------
class ChatRequest(BaseModel):
    organization_id: str
    message: str
    thread_id: Optional[str] = None  # New thread if not provided

class ChatResponse(BaseModel):
    thread_id: str
    history: List[Dict[str, str]]
    reply: str

# --------------------------
# Chat Database Helpers with Redis Caching
# --------------------------
async def store_chat_message(db: AsyncSession, thread_id: str, role: str, message: str):
    from uuid import uuid4
    msg = ChatMessage(
        id=str(uuid4()),
        thread_id=thread_id,
        role=role,
        message=message,
        timestamp=datetime.utcnow()
    )
    db.add(msg)
    await db.commit()
    cache_key = f"chat_history:{thread_id}"
    await redis_client.delete(cache_key)
    logger.info("Stored message for thread %s and invalidated cache.", thread_id)

async def get_chat_history(db: AsyncSession, thread_id: str) -> List[Dict[str, str]]:
    cache_key = f"chat_history:{thread_id}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit for chat history of thread %s.", thread_id)
        return json.loads(cached)
    logger.info("Cache miss for chat history of thread %s. Querying PostgreSQL.", thread_id)
    from sqlalchemy import select
    result = await db.execute(select(ChatMessage).where(ChatMessage.thread_id == thread_id).order_by(ChatMessage.timestamp))
    messages = result.scalars().all()
    MAX_MESSAGES = 50
    trimmed = messages[-MAX_MESSAGES:]
    history = [{"role": msg.role, "message": msg.message} for msg in trimmed]
    await redis_client.set(cache_key, json.dumps(history), ex=60)
    return history

async def create_new_thread(db: AsyncSession, organization_id: str, initial_message: str) -> str:
    thread_id = str(uuid.uuid4())
    thread = ChatThread(
        id=thread_id,
        organization_id=organization_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    db.add(thread)
    await db.commit()
    await store_chat_message(db, thread_id, "user", initial_message)
    logger.info("Created new chat thread %s for organization %s.", thread_id, organization_id)
    return thread_id


# --------------------------
# Helper function to derive knowledge from the knowledgebase
# --------------------------
async def get_knowledge_from_kb(organization_id: str, query: str, top: int = 5) -> list:
    """
    Retrieves relevant knowledge base entries from Qdrant for the given query.
    Returns a list of text strings (or other desired payload field).
    """
    cache_key = f"kb_search:{organization_id}:{query}:{top}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit for KB search query '%s' for organization %s.", query, organization_id)
        return json.loads(cached)

    try:
        query_embedding = await get_embedding(query)
        logger.info(f"Query embedding done")
    except Exception as e:
        logger.error("Error generating embedding for KB query: %s", e)
        return []

    try:
        results = qdrant_client.search(
            collection_name=KNOWLEDGEBASE_COLLECTION,
            query_vector=query_embedding,
            limit=top,
            with_payload=True
        )
        logger.info(f"Qdrant search results: {results}")
    except Exception as e:
        logger.error("Error querying Qdrant for KB: %s", e)
        return []

    # Assume each payload contains a "text" field holding the useful content.
    knowledge_results = [res.payload.get("text_snippet", "") for res in results if res.payload.get("text_snippet")]
    logger.info("Retrieved %d knowledge entries for query '%s' for organization %s.", len(knowledge_results), query, organization_id)
    await redis_client.set(cache_key, json.dumps(knowledge_results), ex=60)
    return knowledge_results


# --------------------------
# Qdrant Query with Caching
# --------------------------
@app.get("/qdrant/search")
async def qdrant_search(organization_id: str, query: str, top: int = 5):
    cache_key = f"qdrant_search:{organization_id}:{query}:{top}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit for Qdrant search query '%s' for organization %s.", query, organization_id)
        return JSONResponse(content=json.loads(cached))
    
    try:
        query_embedding = await get_embedding(query)
        logger.info(f"Query embedding done")
    except Exception as e:
        logger.error("Error generating embedding for query: %s", e)
        raise HTTPException(status_code=500, detail="Embedding error")
    
    try:
        results = qdrant_client.search(
            collection_name=KNOWLEDGEBASE_COLLECTION,
            query_vector=query_embedding,
            limit=top,
            with_payload=True
        )
        logger.info(f"Qdrant search results: {results}")
    except Exception as e:
        logger.error("Error querying Qdrant: %s", e)
        raise HTTPException(status_code=500, detail="Qdrant query error")
    
    results_json = [res.payload for res in results]
    await redis_client.set(cache_key, json.dumps(results_json), ex=60)
    return JSONResponse(content=results_json)


# --------------------------
# Monitoring Endpoint (Example for Prometheus)
# --------------------------
@app.get("/metrics")
async def metrics():
    metrics_data = {
        "uptime_seconds": 12345,  # You can dynamically compute uptime here
        "chat_threads": 42,       # Query from PostgreSQL/statistics
        "qdrant_queries_cached": 10  # Example metric
    }
    return JSONResponse(content=metrics_data)

# --------------------------
# API Endpoints
# --------------------------
@app.post("/upload")
async def upload_file(
    organization_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a file to add to the organization’s knowledgebase.
    Enqueues a Celery task for background processing.
    """
    file_content = await file.read()
    from tasks import process_file_and_ingest_task
    task = process_file_and_ingest_task.delay(organization_id, file_content, file.filename)
    logger.info("Enqueued file upload task for organization %s, file: %s", organization_id, file.filename)
    return {"status": "queued", "task_id": task.id}

@app.post("/chat/new", response_model=ChatResponse)
async def new_chat(chat_request: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Start a new chat thread using OpenAI ChatCompletion for context-aware replies,
    augmented with knowledge from the knowledgebase.
    """
    # Create a new chat thread and store the initial user message.
    thread_id = await create_new_thread(db, chat_request.organization_id, chat_request.message)
    history = await get_chat_history(db, thread_id)

    # Retrieve additional context from the knowledgebase.
    knowledge_results = await get_knowledge_from_kb(chat_request.organization_id, chat_request.message)
    logger.info(f"Knowledge results: {knowledge_results}")
    if knowledge_results:
        # Insert a system message with the relevant knowledge.
        system_context = (
            "Relevant knowledge from the knowledge base:\n" +
            "\n".join(knowledge_results)
        )
        history.insert(0, {"role": "system", "content": system_context})
    
    # Generate a reply using the augmented history.
    reply = await generate_chat_reply(history, chat_request.message)

    logger.info(f"Reply: {reply}")
    await store_chat_message(db, thread_id, "assistant", reply)
    # Retrieve the updated history (with the assistant's reply) for the response.
    logger.info(f"History: {history}")
    history = await get_chat_history(db, thread_id)
    logger.info("Started new chat thread %s for organization %s.", thread_id, chat_request.organization_id)
    return ChatResponse(thread_id=thread_id, history=history, reply=reply)

@app.post("/chat/continue", response_model=ChatResponse)
async def continue_chat(chat_request: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Continue an existing chat thread using OpenAI ChatCompletion,
    augmented with knowledge from the knowledgebase.
    """
    if not chat_request.thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required to continue chat")
    
    # Store the new user message.
    await store_chat_message(db, chat_request.thread_id, "user", chat_request.message)
    history = await get_chat_history(db, chat_request.thread_id)
    
    # Retrieve relevant knowledge for the new message.
    knowledge_results = await get_knowledge_from_kb(chat_request.organization_id, chat_request.message)
    if knowledge_results:
        system_context = (
            "Relevant knowledge from the knowledge base:\n" +
            "\n".join(knowledge_results)
        )
        history.insert(0, {"role": "system", "content": system_context})
    
    # Generate a reply using the augmented conversation history.
    reply = await generate_chat_reply(history, chat_request.message)
    await store_chat_message(db, chat_request.thread_id, "assistant", reply)
    history = await get_chat_history(db, chat_request.thread_id)
    logger.info("Continued chat thread %s for organization %s.", chat_request.thread_id, chat_request.organization_id)
    return ChatResponse(thread_id=chat_request.thread_id, history=history, reply=reply)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# --------------------------
# Application Startup & Shutdown
# --------------------------





# --------------------------
# Main Entry Point
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4, reload=True)

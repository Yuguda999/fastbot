import asyncio
from datetime import datetime
import json
from typing import Dict, List
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from app.config import MODEL_PROVIDER, REDIS_URL
from app.gen_models.gemini_model import generate_title
from app.models import ChatMessage, ChatRequest, ChatResponse, ChatThread, MessageResponse, ThreadHistoryResponse
from app.database import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.qdrant_utils import get_knowledge_from_kb
from app.utils.embedding import get_embedding  # if needed in chat reply generation
from app.utils.text_processing import async_generate_embedding  # if needed
from app.utils import embedding  # adjust if you have a custom generate_reply implementation
import logging
import redis.asyncio as redis


router = APIRouter()
logger = logging.getLogger(__name__)
redis_client = redis.from_url(REDIS_URL, encoding="utf8", decode_responses=True)

if MODEL_PROVIDER.lower() == "openai":
    from app.gen_models.openai_model import generate_reply as generate_reply_openai
elif MODEL_PROVIDER.lower() == "gemini":
    from app.gen_models.gemini_model import generate_reply as generate_reply_gemini
elif MODEL_PROVIDER.lower() == "deepseek":
    from app.gen_models.deepseek_model import generate_reply as generate_reply_deepseek
else:
    raise ValueError("Invalid MODEL_PROVIDER specified in the .env file.")

async def create_new_thread(db: AsyncSession, title:str, organization_id: str, user_id:str, initial_message: str) -> str:
    thread_id = str(uuid.uuid4())
    thread = ChatThread(
        id=thread_id,
        title=title,
        organization_id=organization_id,
        user_id=user_id,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    db.add(thread)
    await db.commit()
    await store_chat_message(db, thread_id, "user", initial_message)
    logger.info("Created new chat thread %s for organization %s.", thread_id, organization_id)
    return thread_id

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

@router.post("/new", response_model=ChatResponse)
async def new_chat(user_id:str, chat_request: ChatRequest, db: AsyncSession = Depends(get_db_session)):
    """
    Start a new chat thread using OpenAI ChatCompletion for context-aware replies,
    augmented with knowledge from the knowledgebase.
    """
    # Create a new chat thread and store the initial user message.
    thread_title_prompt = f"Summarize this chat, in a short, meaningful title (max 20 words).'{chat_request.message}'"
    thread_title = generate_title(thread_title_prompt)
    thread_id = await create_new_thread(db, thread_title, chat_request.organization_id, user_id, chat_request.message)
    history = await get_chat_history(db, thread_id)

    # Retrieve additional context from the knowledgebase.
    knowledge_results = await get_knowledge_from_kb(chat_request.organization_id, chat_request.message, redis_client)
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
    return ChatResponse(thread_id=thread_id, title=thread_title, history=history, reply=reply)

@router.post("/continue", response_model=ChatResponse)
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

@router.get("/threads", response_model=List[ThreadHistoryResponse])
async def get_all_threads(
    organization_id: str = Query(..., description="The organization identifier"),
    user_id: str = Query(..., description="The user identifier"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Retrieve all chat threads and their chat history for the given organization and user.
    """
    # Query threads for the given organization and user.
    threads_stmt = select(ChatThread).where(
        ChatThread.organization_id == organization_id,
        ChatThread.user_id == user_id,
    )
    result = await db.execute(threads_stmt)
    threads = result.scalars().all()

    if not threads:
        raise HTTPException(status_code=404, detail="No threads found for the provided organization and user.")

    thread_histories: List[ThreadHistoryResponse] = []
    # For each thread, query its messages sorted by timestamp.
    for thread in threads:
        msg_stmt = (
            select(ChatMessage)
            .where(ChatMessage.thread_id == thread.id)
            .order_by(ChatMessage.timestamp)
        )
        msg_result = await db.execute(msg_stmt)
        messages = msg_result.scalars().all()

        history = [
            MessageResponse(
                role=msg.role,
                message=msg.message,
                timestamp=msg.timestamp,
            )
            for msg in messages
        ]

        thread_histories.append(
            ThreadHistoryResponse(
                thread_id=thread.id,
                title=thread.title,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                history=history,
            )
        )

    return thread_histories
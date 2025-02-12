import asyncio
from datetime import datetime
from typing import List, Dict
import uuid

from fastapi import UploadFile
from app.config import VECTOR_DIM
from app.gen_models.gemini_model import generate_summary, get_title_and_summary
from app.models import UploadedFile
from app.utils.embedding import get_embedding
from app.utils.file_extraction import extract_text  # Adjust if needed
import logging

from app.utils.qdrant_utils import add_embedding_to_qdrant
from app.database import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("file-chat-api")


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, trying to respect code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        chunk = text[start:end]
        # Try to break at code block, paragraph, or sentence boundary.
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
    Split text into chunks, generate embeddings concurrently,
    and return the average embedding.
    """
    chunks = chunk_text(text)
    if not chunks:
        return [0.0] * VECTOR_DIM
    embeddings = await asyncio.gather(*(get_embedding(chunk) for chunk in chunks))
    # Compute average embedding for each dimension.
    avg_embedding = [float(sum(dim)) / len(dim) for dim in zip(*embeddings)]
    return avg_embedding

async def async_process_text(text: str, filename: str) -> (List[float], Dict[str, str]):  # type: ignore
    """
    Run asynchronous tasks to generate an average embedding and obtain title/summary.
    For title/summary extraction, we send the first chunk.
    """
    embedding = await async_generate_embedding(text)
    chunks = chunk_text(text)
    title_summary = await get_title_and_summary(chunks[0] if chunks else text, filename)
    return embedding, title_summary

def process_file_and_ingest_with_metadata_sync(organisation_id: str, user_id: str, username: str, file: UploadFile) -> str:
    # Extract text from the file.
    text = extract_text(file)
    
    # Generate the summary using Gemini.
    text_prompt = f"Summarize this in one sentence.{text}..."
    summary = generate_summary(text_prompt)
    
    # Process the file for Qdrant ingestion.
    try:
        # Assume async_process_text is defined elsewhere and returns (embedding, title_summary)
        # Wrap the async call in asyncio.run() so that it completes synchronously.
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
    
    # Upsert into Qdrant.
    add_embedding_to_qdrant(organisation_id, doc_id, embedding, metadata)
    
    # Now store file metadata in the database using the synchronous engine.
    db = SessionLocal()  # synchronous session
    try:
        new_file = UploadedFile(
            id=str(uuid.uuid4()),
            filename=file.filename,
            filesize=len(file.file.read()),
            filetype=file.filename.split('.')[-1],
            user_id=user_id,
            username=username,
            organisation_id=organisation_id,
            summary=summary,
            upload_date=datetime.utcnow(),
            timestamp=int(datetime.utcnow().timestamp())
        )
        db.add(new_file)
        db.commit()
        db.refresh(new_file)
    except Exception as e:
        logger.error("Error storing file metadata: %s", e)
        db.rollback()
    finally:
        db.close()
    
    return doc_id
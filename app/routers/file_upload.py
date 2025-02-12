from typing import List, Optional
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, BackgroundTasks
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import UploadedFile, FileResponse
from app.database import get_db_session
from app.models import FileResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload")
async def upload_file(
    organisation_id: str = Form(...),
    user_id: str = Form(...),
    username: str = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a file. The file is processed to extract text, generate a summary,
    upsert embeddings into Qdrant, and file metadata is stored in the database.
    """
    file_content = await file.read()
    # Enqueue a Celery task to process the file in the background.
    from app.tasks import process_file_and_ingest_with_metadata_task
    task = process_file_and_ingest_with_metadata_task.delay(organisation_id, user_id, username, file_content, file.filename)
    logger.info("Enqueued file upload for org %s, file: %s", organisation_id, file.filename)
    return {"status": "queued", "task_id": task.id}

@router.get("/", response_model=List[FileResponse])
async def get_files(
    organisation_id: str = Query(..., description="Organization identifier"),
    user_id: Optional[str] = Query(None, description="User identifier (optional)"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve file metadata for a given organization and (optionally) user.
    """
    stmt = select(UploadedFile).where(UploadedFile.organisation_id == organisation_id)
    if user_id:
        stmt = stmt.where(UploadedFile.user_id == user_id)
    
    result = await db.execute(stmt)
    files = result.scalars().all()
    return files

import io
from fastapi.datastructures import UploadFile
from app.celery_app import celery_app  
from app.routers.scrape import scrape_urls_task
from app.utils.text_processing import process_file_and_ingest_with_metadata_sync
import asyncio
from app.celery_app import celery_app

@celery_app.task
def process_file_and_ingest_with_metadata_task(organisation_id: str, user_id: str, username: str, file_content: bytes, filename: str):
    """
    Celery task to process an uploaded file:
      - Extract text and generate summary.
      - Process file for Qdrant ingestion.
      - Store file metadata in the database.
    """
    # Create an UploadFile-like object from the file content.
    file_obj = UploadFile(filename=filename, file=io.BytesIO(file_content))
    return process_file_and_ingest_with_metadata_sync(organisation_id, user_id, username, file_obj)

@celery_app.task
def scrape_urls_celery(organization_id: str, urls: list, max_concurrent: int = 5):
    """
    Celery task to run the asynchronous URL scraping task.
    This wraps the async function using asyncio.run() so it can be executed in a Celery worker.
    """
    asyncio.run(scrape_urls_task(organization_id, urls, max_concurrent))
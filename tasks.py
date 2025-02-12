# import io
# from app.utils.text_processing import process_file_and_ingest_sync
# from celery_app import celery_app

# @celery_app.task
# def process_file_and_ingest_task(organization_id: str, user_id: str, username: str, file_content: bytes, filename: str):
#     """
#     Celery task to process a file:
#       - Extract text and chunk it
#       - Generate embeddings and obtain title/summary via OpenAI
#       - Upsert document (or document chunks) into Qdrant
#     """
#     from fastapi.datastructures import UploadFile
#     file_obj = UploadFile(filename=filename, file=io.BytesIO(file_content))
#     return process_file_and_ingest_sync(organization_id, file_obj)

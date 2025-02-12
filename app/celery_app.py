import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve broker and backend URLs from environment variables.
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create the Celery application.
# The "include" option ensures that Celery automatically imports the specified modules,
# registering any tasks found in those modules.
celery_app = Celery(
    "fastbot",
    broker=broker_url,
    backend=result_backend,
    include=["app.tasks"]
)

# Optional: Update Celery configuration.
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],  # Only accept JSON content.
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

if __name__ == "__main__":
    # For debugging purposes, print out configuration information.
    print("Broker URL:", broker_url)
    print("Result Backend:", result_backend)

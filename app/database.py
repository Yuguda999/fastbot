import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql+asyncpg://user:pass@host/dbname"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")

# Create the asynchronous engine.
async_engine = create_async_engine(DATABASE_URL, echo=False)

# Create an async sessionmaker using AsyncSession.
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,  # Use the actual AsyncSession class.
    expire_on_commit=False
)

# For synchronous operations in Celery tasks, create a synchronous engine.
# Replace "+asyncpg" with the appropriate sync driver, e.g., remove it for psycopg2.
sync_database_url = DATABASE_URL.replace("+asyncpg", "")
from sqlalchemy import create_engine
sync_engine = create_engine(sync_database_url, echo=False)

# Synchronous sessionmaker.
SessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)

Base = declarative_base()
# Dependency to get database session
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
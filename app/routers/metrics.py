from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/metrics")
async def metrics():
    metrics_data = {
        "uptime_seconds": 12345,  # Compute dynamically if needed.
        "chat_threads": 42,       # Replace with actual query results.
        "qdrant_queries_cached": 10  # Example metric.
    }
    return JSONResponse(content=metrics_data)

import json
import logging
from app.config import VECTOR_DIM, QDRANT_HOST, QDRANT_PORT
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from app.utils.embedding import get_embedding
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_qdrant_collection(organization_id: str):
    """Ensure Qdrant collection exists or create it."""
    knowledgebase_collection = f"kb_{organization_id}"
    try:
        _ = qdrant_client.get_collection(collection_name=knowledgebase_collection)
        logger.info("Qdrant collection '%s' exists.", knowledgebase_collection)
    except Exception as e:
        logger.warning("Collection not found; creating collection: %s", e)
        qdrant_client.recreate_collection(
            collection_name=knowledgebase_collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        logger.info("Qdrant collection '%s' created.", knowledgebase_collection)
    finally:
        return knowledgebase_collection

def add_embedding_to_qdrant(organization_id: str, doc_id: str, embedding: list, metadata: dict):
    try:
        knowledgebase_collection = ensure_qdrant_collection(organization_id)
        logger.info(f"Upserting document {doc_id} into Qdrant for org {organization_id} with knowledgebase {knowledgebase_collection}.")

        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"organization_id": organization_id, **metadata}
        )
        qdrant_client.upsert(
            collection_name=knowledgebase_collection,
            points=[point]
        )
        logger.info("Upserted document %s for organization %s.", doc_id, organization_id)
    except Exception as e:
        logger.error("Error upserting into Qdrant: %s", e)
        raise

async def get_knowledge_from_kb(organization_id: str, query: str, redis_client, top: int = 5) -> list:
    """
    Retrieve relevant knowledge entries from Qdrant.
    Assumes each point payload has a 'text_snippet' field.
    """
    knowledgebase_collection = ensure_qdrant_collection(organization_id)
    logger.info(f"Searching KB for query '{query}' in org {organization_id} of knowledgebase {knowledgebase_collection}.")
    cache_key = f"kb_search:{organization_id}:{query}:{top}"
    cached = await redis_client.get(cache_key)
    if cached:
        logger.info("KB cache hit for query '%s' org %s.", query, organization_id)
        return json.loads(cached)
    
    try:
        query_embedding = await get_embedding(query)
    except Exception as e:
        logger.error("Error generating embedding for KB query: %s", e)
        return []
    
    try:
        results = qdrant_client.search(
            collection_name=knowledgebase_collection,
            query_vector=query_embedding,
            limit=top,
            with_payload=True
        )
    except Exception as e:
        logger.error("Error querying Qdrant for KB: %s", e)
        return []
    
    knowledge_results = [res.payload.get("text_snippet", "") for res in results if res.payload.get("text_snippet")]
    await redis_client.set(cache_key, json.dumps(knowledge_results), ex=60)
    return knowledge_results

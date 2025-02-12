import logging
from typing import List
from langchain_ollama import OllamaEmbeddings
from openai import AsyncOpenAI
import asyncio
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="intfloat/e5-large-v2",
    token=os.getenv("HUGGINGFACE_TOKEN"))


logger = logging.getLogger(__name__)

async def get_embedding(text: str) -> List[float]:
    """
    Asynchronously get an embedding vector from Hugging Face's E5-large-v2 model.
    
    Parameters:
      - text: The input string for which the embedding is generated.
      
    Returns:
      - A list of floats representing the embedding.
      - On error, returns a zero vector with length 1536.
    """
    try:
        # Since client.feature_extraction is a blocking call, we run it in a separate thread.
        result = await asyncio.to_thread(client.feature_extraction, text)
        
        # The result is typically a NumPy array. Convert it to a list if possible.
        if hasattr(result, "tolist"):
            return result.tolist()
        else:
            return list(result)
    except Exception as e:
        logger.error("Error getting embedding: %s", e)
        return [0.0] * 1024  # Adjust VECTOR_DIM if necessary


async def example():
    embedding = await get_embedding("Hi, who are you?")
    print("Embedding vector shape:", len(embedding))

if __name__ == "__main__":
    import asyncio
    asyncio.run(example())

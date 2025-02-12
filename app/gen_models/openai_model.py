import os
import json
import asyncio
import logging
from typing import Dict, List
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Asynchronously extract title and summary from a text chunk using OpenAI.
    Uses a system prompt to instruct the LLM.
    """
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks. "
        "Return a JSON object with 'title' and 'summary' keys. "
        "For the title: If this seems like the start of a document, extract its title. "
        "If it's a middle chunk, derive a descriptive title. "
        "For the summary: Create a concise summary of the main points in this chunk. "
        "Keep both concise but informative."
    )

    try:
        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error("Error getting title and summary: %s", e)
        return {"title": "Error processing title", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Asynchronously get an embedding vector from OpenAI."""
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Error getting embedding: %s", e)
        return [0.0] * 1536  # Adjust VECTOR_DIM if necessary


async def generate_reply(messages: List[Dict[str, str]], model="gpt-4o-mini") -> str:
    """
    Generate a reply using OpenAI's chat completion endpoint.
    Parameters:
      - messages: a list of dictionaries, each with 'role' and 'content'
      - model: model name (default: "gpt-4o-mini")
    Returns:
      - The reply as a string.
    """
    try:
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Error generating reply with OpenAI: %s", e)
        return "I'm sorry, I'm having trouble generating a response right now."

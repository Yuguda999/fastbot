import json
import os
from typing import Dict, List
import google.generativeai as genai
import logging
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger("gemini_model")

# Configure the Gemini client using the API key from the environment.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def generate_reply(prompt, model="gemini-1.5-flash"):
    """
    Generate content using Gemini.
    
    Parameters:
      - prompt: a string prompt that includes the conversation context.
      - model: model name (default: "gemini-1.5-flash")
    
    Returns:
      - The generated text as a string.
    """
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error("Error generating reply with Gemini: %s", e)
        return "I'm sorry, I'm having trouble generating a response right now."

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Asynchronously extract title and summary from a text chunk using Google's Gemini model.
    """
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks. "
        "Return a JSON object with 'title' and 'summary' keys. "
        "For the title: If this is the start of a document, extract its title. "
        "If it's a middle chunk, derive a descriptive title. "
        "For the summary: Create a concise summary of the main points in this chunk. "
        "Keep both concise but informative."
    )

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            [
                {"text": system_prompt},
                {"text": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ]
        )
        
        # Extract and parse response text
        content = response.text.strip()
        return json.loads(content) if content.startswith("{") else {"title": "N/A", "summary": content}
    
    except Exception as e:
        logger.error("Error getting title and summary with Gemini: %s", e)
        return {"title": "Error processing title", "summary": "Error processing summary"}
    

def generate_title(prompt, model="gemini-1.5-flash"):
    """
    Generate a thread title using Gemini.
    
    Parameters:
      - prompt: A string prompt providing context for title generation.
      - model: The model name to use (default: "gemini-1.5-flash").
      
    Returns:
      - The generated title as a string.
    """
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error("Error generating title with Gemini: %s", e)
        return "Untitled Thread"

def generate_summary(text: str, model: str = "gemini-1.5-flash") -> str:
    """
    Generate a summary of the provided text using the Gemini model.
    
    Parameters:
      - text: The text to summarize.
      - model: The Gemini model to use (default: "gemini-1.5-flash").
    
    Returns:
      - The generated summary as a string.
    """
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(text)
        return response.text.strip()
    except Exception as e:
        logger.error("Error generating summary with Gemini: %s", e)
        return "Summary not available."

# Set model name and dimensionality
MODEL_NAME = "multilingual-e5" # or "text-embedding-005"
DIMENSIONALITY = 1536  

async def get_embedding_gemini(text: str) -> list[float]:
    """Asynchronously get an embedding vector from Google's embedding model with 1536 dimensions."""
    try:
        # Initialize the model
        model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        
        # Create the input for embedding
        input_text = [TextEmbeddingInput(text, task="TEXT_EMBEDDING")]
        
        # Get embeddings with the specified dimensionality
        embeddings = model.get_embeddings(input_text, output_dimensionality=DIMENSIONALITY)
        
        # Extract the embedding values from the response
        return embeddings[0].values  # Assuming the response returns embeddings as a list of embeddings
    except Exception as e:
        logger.error("Error getting embedding with Gemini: %s", e)
        return [0.0] * DIMENSIONALITY  # Return a zero vector on error

async def get_embedding_multilingual_e5(text: str) -> list[float]:
    """Asynchronously get an embedding vector from the Multilingual E5 model with 1536 dimensions."""
    try:
        # Initialize the Multilingual E5 model
        model = TextEmbeddingModel.from_pretrained(MODEL_NAME)
        
        # Create the input for embedding
        input_text = [TextEmbeddingInput(text, task="TEXT_EMBEDDING")]
        
        # Get embeddings with the specified dimensionality
        embeddings = model.get_embeddings(input_text, output_dimensionality=DIMENSIONALITY)
        
        # Extract the embedding values from the response
        return embeddings[0].values  # Assuming the response returns embeddings as a list of embeddings
    except Exception as e:
        logger.error("Error getting embedding with Multilingual E5: %s", e)
        return [0.0] * DIMENSIONALITY  # Return a zero vector on error


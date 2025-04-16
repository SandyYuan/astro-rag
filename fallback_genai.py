"""
Fallback implementation for Google Generative AI API.
This provides a minimal implementation of the Google GenerativeAI client 
using direct HTTP requests when the official client isn't available.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, List, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
_CONFIG = {
    "api_key": None,
    "base_url": "https://generativelanguage.googleapis.com/v1",
}

class SimpleResponse:
    """Simple class to mimic the response object from the official client"""
    
    def __init__(self, text: str, raw_response: Dict[str, Any] = None):
        self.text = text
        self.raw_response = raw_response
    
    def __str__(self):
        return self.text

class GenerativeModel:
    """Simple implementation of GenerativeModel class"""
    
    def __init__(self, model_name: str, generation_config: Union[Dict[str, Any], float] = None, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or _CONFIG["api_key"]
        
        # Handle different types of generation_config
        if generation_config is None:
            self.generation_config = {}
        elif isinstance(generation_config, float):
            # Handle case where temperature is passed directly
            self.generation_config = {"temperature": generation_config}
        else:
            self.generation_config = generation_config
        
        if not self.api_key:
            raise ValueError("API key must be provided either through configure() or directly to GenerativeModel")
        
        logger.info(f"Initialized fallback GenerativeModel with model: {model_name}")
    
    def generate_content(self, prompt: str) -> SimpleResponse:
        """Generate content using HTTP request to Google Gemini API"""
        url = f"{_CONFIG['base_url']}/models/{self.model_name}:generateContent"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": self.generation_config
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        try:
            logger.info(f"Sending request to {url}")
            response = requests.post(url, json=payload, headers=headers)
            
            # Handle HTTP errors
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            logger.debug(f"Response received: {data}")
            
            # Extract the text from the response
            try:
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return SimpleResponse(text=text, raw_response=data)
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing response: {e}")
                return SimpleResponse(text="Error: Unable to parse response from Gemini API", raw_response=data)
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return SimpleResponse(text=f"Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return SimpleResponse(text=f"Error: {str(e)}")

class EmbeddingResponse:
    """Simple class to hold embedding response data"""
    
    def __init__(self, values: List[List[float]]):
        self.values = values

def embed_content(
    content: Union[str, List[str]], 
    model: str = "models/text-embedding-004", 
    task_type: str = "RETRIEVAL_DOCUMENT",
    api_key: str = None
) -> Union[List[float], List[List[float]]]:
    """Embed content using Google's embedding API
    
    Args:
        content: String or list of strings to embed
        model: Embedding model to use
        task_type: Type of embedding (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)
        api_key: Google API key (optional if already configured)
        
    Returns:
        For single input: List of floats representing the embedding
        For multiple inputs: List of list of floats representing the embeddings
    """
    api_key = api_key or _CONFIG["api_key"]
    if not api_key:
        raise ValueError("API key must be provided either through configure() or directly")
    
    url = f"{_CONFIG['base_url']}/{model}:embedContent"
    
    # Handle single string vs list of strings
    is_single = isinstance(content, str)
    contents = [content] if is_single else content
    
    payload = {
        "model": model,
        "content": {"parts": [{"text": text} for text in contents]},
        "taskType": task_type
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    try:
        logger.info(f"Sending embedding request to {url}")
        response = requests.post(url, json=payload, headers=headers)
        
        # Handle HTTP errors
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        logger.debug(f"Embedding response received")
        
        # Extract the embeddings from the response
        try:
            embeddings = data.get("embeddings", [])
            if not embeddings:
                logger.warning(f"No embeddings found in response: {data}")
                return [] if is_single else [[]]
            
            # Extract the values from each embedding
            values = [embedding.get("values", []) for embedding in embeddings]
            
            # Return single embedding for single input, list of embeddings for multiple inputs
            if is_single:
                return values[0] if values else []
            else:
                return values
        except Exception as e:
            logger.error(f"Error parsing embedding response: {e}")
            if is_single:
                return []
            else:
                return [[] for _ in range(len(contents))]
                
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error in embedding: {e.response.status_code} - {e.response.text}")
        if is_single:
            return []
        else:
            return [[] for _ in range(len(contents))]
    except Exception as e:
        logger.error(f"Error in embedding: {e}")
        if is_single:
            return []
        else:
            return [[] for _ in range(len(contents))]

def batch_embed_contents(texts: List[str], **kwargs) -> List[List[float]]:
    """Helper function to batch embed multiple texts
    Useful for compatibility with LangChain's Embeddings interface
    """
    return embed_content(content=texts, **kwargs)

def configure(api_key: str = None, **kwargs):
    """Configure the fallback client with API key and other parameters"""
    if api_key:
        _CONFIG["api_key"] = api_key
    
    # Add any other configuration parameters
    for key, value in kwargs.items():
        if key in _CONFIG:
            _CONFIG[key] = value
    
    logger.info("Configured fallback GenerativeAI client") 
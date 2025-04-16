import os
import logging
from typing import Optional, Dict, Any, List

# LangChain imports for compatibility
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
# Import conditionally handled in class implementation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for Google's Gemini models"""
    
    def __init__(self, api_key: str):
        """Initialize the Google Gemini client
        
        Args:
            api_key: Google API key for Gemini models
        """
        if not api_key:
            raise ValueError("API key is required for Gemini")
            
        self.api_key = api_key
        self.genai = None
        
        # Try multiple approaches to import the google.generativeai package
        error_messages = []
        
        # Approach 1: Standard import
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            logger.info("Initialized Gemini client successfully using direct import")
            return
        except ImportError as e:
            error_messages.append(f"Standard import failed: {str(e)}")
        except Exception as e:
            error_messages.append(f"Error during standard import: {str(e)}")
        
        # Approach 2: Try alternate import path
        try:
            from google import generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            logger.info("Initialized Gemini client successfully using alternate import")
            return
        except ImportError as e:
            error_messages.append(f"Alternate import failed: {str(e)}")
        except Exception as e:
            error_messages.append(f"Error during alternate import: {str(e)}")
        
        # Approach 3: Try dynamic installation
        try:
            import sys, subprocess
            logger.info("Attempting to install google-generativeai package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "google-generativeai==0.3.2"])
            
            # Try import again after installation
            import importlib
            importlib.invalidate_caches()
            
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            logger.info("Initialized Gemini client successfully after dynamic installation")
            return
        except Exception as e:
            error_messages.append(f"Dynamic installation failed: {str(e)}")
        
        # If we reach here, all approaches failed
        detailed_error = "\n".join(error_messages)
        logger.error(f"All import approaches failed: {detailed_error}")
        raise ImportError(f"Failed to import google.generativeai. Tried multiple approaches: {detailed_error}")
    
    def generate_content(self, prompt: str, temperature: float = 0.7, model_name: str = "gemini-2.5-pro-exp-03-25") -> str:
        """Generate content using Gemini
        
        Args:
            prompt: The text prompt to send to Gemini
            temperature: Controls randomness (0 = deterministic, 1 = creative)
            model_name: Gemini model to use
            
        Returns:
            Generated text response
        """
        if not self.genai:
            raise ValueError("Gemini client not properly initialized")
            
        try:
            logger.debug(f"Generating content with model {model_name}")
            
            # Create the model with the appropriate configuration
            model = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config={"temperature": temperature}
            )
            
            # Generate content
            response = model.generate_content(prompt)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

class LLMClientWrapper(LLM):
    """LangChain-compatible wrapper for the Gemini client"""
    
    client: LLMClient
    temperature: float = 0.3
    model_name: str = "gemini-2.5-pro-exp-03-25"
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the Gemini API with the given prompt"""
        return self.client.generate_content(
            prompt, 
            temperature=self.temperature,
            model_name=self.model_name
        )

class LLMEmbeddings(Embeddings):
    """LangChain-compatible embeddings using Google's embedding models"""
    
    def __init__(self, google_api_key: str, model: str = "models/text-embedding-004"):
        """Initialize the embeddings with Google API key
        
        Args:
            google_api_key: Google API key
            model: Embedding model name
        """
        if not google_api_key:
            raise ValueError("Google API key is required for embeddings")
        
        self.api_key = google_api_key
        self.model = model
        self.embeddings = None
        
        # Try to use official LangChain integration
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=model, 
                google_api_key=google_api_key
            )
            logger.info(f"Initialized Google embeddings with model: {model} (official LangChain integration)")
            return
        except Exception as e:
            logger.warning(f"Could not use LangChain integration for embeddings: {e}")
            raise ValueError(f"Failed to initialize embeddings: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.embeddings:
            return self.embeddings.embed_documents(texts)
        else:
            raise ValueError("Embeddings not properly initialized")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        if self.embeddings:
            return self.embeddings.embed_query(text)
        else:
            raise ValueError("Embeddings not properly initialized")

class LLMProvider:
    """
    Provider for Google's Gemini models and embeddings
    """
    
    # Default model configuration
    PROVIDER_GOOGLE = "google"  # Kept for backward compatibility
    DEFAULT_EMBEDDING_MODEL = "models/text-embedding-004"
    DEFAULT_TEXT_MODEL = "gemini-2.5-pro-exp-03-25"
    
    def __init__(self, api_key: str = None, provider: str = None, embedding_model: str = None):
        """
        Initialize the Gemini provider with API key
        
        Args:
            api_key: Google API key (required)
            provider: Ignored parameter (kept for backward compatibility)
            embedding_model: Embedding model name (optional)
        """
        # Handle API key from parameter or environment
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Either provide it directly or set GOOGLE_API_KEY environment variable.")
        
        # Set default models
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL
        
        # Initialize the Gemini client
        try:
            self.client = LLMClient(api_key=self.api_key)
            logger.info(f"Initialized Gemini provider")
            logger.info(f"Using embedding model: {self.embedding_model}")
        except ImportError as e:
            logger.error(f"Error initializing LLMClient: {e}")
            raise ValueError(f"Unable to initialize LLMClient: {str(e)}")
    
    def get_llm(self, **kwargs) -> LLM:
        """
        Get a LangChain-compatible LLM instance
        
        Args:
            **kwargs: Additional arguments including:
                - temperature: Controls randomness (0-1)
                - model_name: Gemini model to use
                
        Returns:
            A LangChain-compatible LLM instance
        """
        temperature = kwargs.pop("temperature", 0.3)
        model_name = kwargs.pop("model_name", self.DEFAULT_TEXT_MODEL)
        
        logger.info(f"Creating LLM with model: {model_name}, temperature: {temperature}")
        return LLMClientWrapper(
            client=self.client, 
            temperature=temperature,
            model_name=model_name
        )
    
    def get_embeddings(self, **kwargs) -> Embeddings:
        """
        Get a LangChain-compatible embeddings instance
        
        Args:
            **kwargs: Additional arguments (unused)
            
        Returns:
            A LangChain-compatible embeddings instance
        """
        logger.info(f"Creating embeddings with model: {self.embedding_model}")
        return LLMEmbeddings(
            google_api_key=self.api_key, 
            model=self.embedding_model
        )
        
    @property
    def embedding_model_name(self) -> str:
        """Property to access the embedding model name"""
        return self.embedding_model 
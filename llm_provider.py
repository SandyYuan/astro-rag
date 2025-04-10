import os
import logging
from typing import Optional, Dict, Any, List

# LangChain imports for compatibility
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import the provided LLMClient class
class LLMClient:
    """Wrapper for LLM clients to provide a consistent interface"""
    
    def __init__(self, api_key: str, provider: str = "azure"):
        """Initialize the LLM client with the appropriate provider
        
        Args:
            api_key: API key for the selected provider
            provider: 'azure', 'google', or 'claude'
        """
        self.api_key = api_key
        self.provider = provider
        
        if provider == "google":
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("googleai is not installed")
        elif provider == "azure":
            try:
                from langchain_openai import AzureChatOpenAI
                # Hard-coded Azure configuration
                self.client = AzureChatOpenAI(
                    azure_endpoint="https://utbd-omodels-advanced.openai.azure.com",
                    azure_deployment="o1",
                    api_version="2025-01-01-preview",
                    api_key=api_key
                )
            except ImportError:
                raise ImportError("langchain_openai is not installed")
        elif provider == "claude":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic is not installed")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_content(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate content using the configured LLM
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for generation
            
        Returns:
            Generated text response
        """
        if self.provider == "google":
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp", 
                contents=prompt
            )
            return response.text
        elif self.provider == "azure":
            # For Azure, we can directly invoke the client
            response = self.client.invoke(prompt)
            return response.content
        elif self.provider == "claude":
            # For Claude, we need to structure the message differently
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            return response.content[0].text
        
        # Fallback (should never reach here)
        raise ValueError(f"Unsupported provider: {self.provider}")

# Create a LangChain-compatible wrapper for LLMClient
class LLMClientWrapper(LLM):
    """LangChain-compatible wrapper for the LLMClient."""
    
    client: LLMClient
    temperature: float = 0.3
    
    @property
    def _llm_type(self) -> str:
        return f"llm_client_{self.client.provider}"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the LLM client with the given prompt."""
        return self.client.generate_content(prompt, temperature=self.temperature)

# LangChain-compatible embeddings class that uses Google's embeddings
class LLMEmbeddings(Embeddings):
    """LangChain-compatible embeddings that use Google's embeddings API."""
    
    def __init__(self, google_api_key: str, model: str = "models/text-embedding-004"): # gemini-embedding-exp-03-07
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed the given texts."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed the given query."""
        return self.embeddings.embed_query(text)

# Main provider class that manages the LLM and embeddings
class LLMProvider:
    """
    Provider class for Language Models and Embeddings using LLMClient.
    """
    
    # Available provider options
    PROVIDER_GOOGLE = "google"
    PROVIDER_AZURE = "azure"
    PROVIDER_CLAUDE = "claude"
    
    # Default provider configuration
    DEFAULT_PROVIDER = PROVIDER_GOOGLE
    
    def __init__(self, provider=None):
        """
        Initialize the LLM provider.
        
        Args:
            provider (str, optional): LLM provider name. Defaults to environment variable or google.
        """
        # Get provider from environment or use default
        self.provider = provider or os.environ.get("LLM_PROVIDER", self.DEFAULT_PROVIDER)
        
        # Get the appropriate API key
        self.api_key = self._get_api_key()
        
        # Initialize the client
        self.client = LLMClient(api_key=self.api_key, provider=self.provider)
        
        logging.info(f"Initialized LLM provider: {self.provider}")
    
    def _get_api_key(self) -> str:
        """Get the API key for the selected provider."""
        if self.provider == self.PROVIDER_GOOGLE:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            return api_key
        elif self.provider == self.PROVIDER_AZURE:
            api_key = os.environ.get("AZURE_API_KEY")
            if not api_key:
                raise ValueError("AZURE_API_KEY environment variable not set")
            return api_key
        elif self.provider == self.PROVIDER_CLAUDE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            return api_key
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_llm(self, **kwargs) -> LLM:
        """
        Get a LangChain-compatible LLM instance.
        
        Args:
            **kwargs: Additional arguments to pass to the LLM constructor.
            
        Returns:
            A LangChain-compatible LLM instance.
        """
        temperature = kwargs.pop("temperature", 0.3)
        return LLMClientWrapper(client=self.client, temperature=temperature)
    
    def get_embeddings(self, **kwargs) -> Embeddings:
        """
        Get a LangChain-compatible embeddings instance.
        
        Args:
            **kwargs: Additional arguments to pass to the embeddings constructor.
            
        Returns:
            A LangChain-compatible embeddings instance.
        """
        # Currently using Google's embedding model for all providers
        # This can be expanded in the future to support other embedding models
        return LLMEmbeddings(google_api_key=os.environ.get("GOOGLE_API_KEY")) 
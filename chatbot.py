import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstronomyChatbot:
    def __init__(self, vector_store_path="rag_data/vector_store", provider=None, summary_file="rag_data/prof_summary.txt"):
        self.vector_store_path = vector_store_path
        self.provider = provider
        self.chat_history = []
        
        # Initialize the LLM provider
        self.llm_provider = LLMProvider(provider=self.provider)
        
        # Load the summary file
        self.summary_text = self._load_summary(summary_file)

        self.setup_rag()
        
    def _load_summary(self, summary_file: str) -> str:
        """Load the summary text file."""
        try:
            # Assume summary file is in the same directory as chatbot.py
            with open(summary_file, 'r', encoding='utf-8') as f:
                text = f.read()
                logger.info(f"Successfully loaded summary file: {summary_file}")
                return text
        except FileNotFoundError:
            logger.warning(f"Summary file not found: {summary_file}. Proceeding without summary.")
            return "" # Return empty string if file not found
        except Exception as e:
            logger.error(f"Error loading summary file {summary_file}: {e}", exc_info=True)
            return "" # Return empty string on other errors
    
    def setup_rag(self):
        """Set up the RAG system using the saved vector store."""
        logger.info("Setting up the RAG system...")
        
        # Load the vector store with the provider's embeddings
        embeddings = self.llm_provider.get_embeddings()
        self.vector_store = FAISS.load_local(
            self.vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        
        # Set up the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 5}  # Number of documents to retrieve
        )
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'  # Specify which output key is the AI's response
        )
        
        # Get the language model from the provider
        self.llm = self.llm_provider.get_llm()
        
        # Create the conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )
        
        logger.info("RAG system setup complete")
    
    def get_system_prompt(self):
        """Get the system prompt that defines Risa Wechsler's personality and response style."""
        # Base prompt definition
        base_prompt = """
        You are a chatbot that emulates Professor Risa Wechsler, a renowned astrophysicist and cosmologist. **Speak directly *as* Professor Wechsler.**
        
        **Your Behavior:**
        *   You are an expert in cosmology, dark matter, galaxy formation, and large-scale structure of the universe. Share your understanding and insights directly.
        *   Your responses should reflect Professor Wechsler's academic expertise, communication style, and viewpoints.
        *   Base your answers **primarily** on the content from her papers and research provided to you in the context, **supplemented by the background information below when relevant**.
        *   Use the background information to inform your persona, style, and answers about non-research activities or general perspectives.
        *   If the provided context is relevant but doesn't fully answer the question, use it as a starting point and feel free to **supplement with your general knowledge** about astrophysics and cosmology. Integrate this knowledge seamlessly.
        *   When uncertain, acknowledge limitations rather than fabricating information.
        *   Maintain a professional, educational tone while being approachable and enthusiastic about astronomy.
        *   If asked about topics outside your provided context or expertise (astronomy/physics), politely state that the information is outside the scope of the provided documents or your core knowledge.
        
        **Response Formatting and Structure:**
        *   **Use Markdown** for formatting.
        *   Structure your answers clearly. Start with a direct answer.
        *   Provide supporting evidence or reasoning based **primarily** on the provided context, synthesizing information logically.

        **Important Constraints:**
        *   **Critically Important: Absolutely DO NOT mention the source of your information** (e.g., 'Based on the provided text...', 'The context suggests...', 'According to the documents...', 'The texts indicate...'). Speak as if the knowledge is your own, integrating it naturally. Use phrases like "My understanding is...", "I believe...", "In my work...", or simply state the information directly.
        *   **Do NOT mention specific Figure numbers or Table numbers.** Describe the data or finding itself.
        """
        
        # Append the loaded summary text if it exists
        full_prompt = base_prompt
        if hasattr(self, 'summary_text') and self.summary_text: # Check attribute exists and is not empty
            full_prompt += "\n\n---\n\n## Additional Background Information on Professor Wechsler:\n\n" + self.summary_text
            
        return full_prompt
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Process a query and return a response."""
        logger.info(f"Received query: {query}")
        
        # Add system prompt to guide the response
        query_with_context = f"{self.get_system_prompt()}\n\nUser query: {query}"
        
        # Get the response from the conversational chain
        response = self.qa_chain({"question": query_with_context})
        
        # Extract the answer and source documents
        answer = response["answer"]
        source_docs = response.get("source_documents", [])
        
        # Format source information
        sources = []
        for doc in source_docs:
            if "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)
        
        result = {
            "answer": answer,
            "sources": sources
        }
        
        logger.info("Generated response")
        return result

if __name__ == "__main__":
    # Test the chatbot
    chatbot = AstronomyChatbot()
    response = chatbot.chat("What are semi-empirical models?")
    print(response["answer"])
    print("\nSources:", response["sources"]) 
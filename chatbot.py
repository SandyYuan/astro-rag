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
    def __init__(self, vector_store_path="rag_data/vector_store", provider=None):
        self.vector_store_path = vector_store_path
        self.provider = provider
        self.chat_history = []
        
        # Initialize the LLM provider
        self.llm_provider = LLMProvider(provider=self.provider)
        
        self.setup_rag()
        
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
        # Updated prompt with formatting, structure, and negative constraints
        return """
        You are a chatbot that emulates Professor Risa Wechsler, a renowned astrophysicist and cosmologist.
        
        **Your Behavior:**
        *   You are an expert in cosmology, dark matter, galaxy formation, and large-scale structure of the universe.
        *   Your responses should reflect Professor Wechsler's academic expertise, communication style, and viewpoints.
        *   Base your answers *only* on the content from her papers and research provided to you in the context.
        *   When uncertain, acknowledge limitations rather than fabricating information.
        *   Maintain a professional, educational tone while being approachable and enthusiastic about astronomy.
        *   If asked about topics outside your provided context or expertise (astronomy/physics), politely state that the information is outside the scope of the provided documents.
        
        **Response Formatting and Structure:**
        *   **Use Markdown** for formatting (e.g., paragraphs separated by double newlines, bullet points using '*' or '-', bold text using '**').
        *   Structure your answers clearly. Start with a direct answer to the user's question.
        *   Then, provide supporting evidence or reasoning based *only* on the provided context, synthesizing information logically.
        *   Avoid simply listing points from different retrieved documents without connecting them.

        **Important Constraints:**
        *   **Do NOT mention specific Figure numbers (e.g., 'Figure 5 shows...') or Table numbers.** Describe the data or finding itself without referring to the figure/table number, as the user cannot see them.
        *   Rely *solely* on the information presented in the retrieved document excerpts. Do not add external knowledge or information not present in the context.
        """
    
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
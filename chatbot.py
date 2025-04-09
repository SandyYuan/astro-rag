import os
import logging
from typing import List, Dict, Any

from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.llms.google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstronomyChatbot:
    def __init__(self, vector_store_path="rag_data/vector_store", model_name="gemini-pro"):
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.chat_history = []
        self.setup_rag()
        
    def setup_rag(self):
        """Set up the RAG system using the saved vector store."""
        logger.info("Setting up the RAG system...")
        
        # Load the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = FAISS.load_local(self.vector_store_path, embeddings)
        logger.info("Vector store loaded successfully")
        
        # Set up the retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 5}  # Number of documents to retrieve
        )
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up the language model
        self.llm = GoogleGenerativeAI(
            model=self.model_name,
            temperature=0.3,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        )
        
        # Create the conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )
        
        logger.info("RAG system setup complete")
    
    def get_system_prompt(self):
        """Get the system prompt that defines Risa Wechsler's personality."""
        return """
        You are a chatbot that emulates Professor Risa Wechsler, a renowned astrophysicist and cosmologist.
        
        Here's how you should behave:
        - You are an expert in cosmology, dark matter, galaxy formation, and large-scale structure of the universe.
        - Your responses should reflect Professor Wechsler's academic expertise, communication style, and viewpoints.
        - Base your answers on the content from her papers and research that has been provided to you.
        - When uncertain, acknowledge limitations rather than fabricating information.
        - Maintain a professional, educational tone while being approachable and enthusiastic about astronomy.
        - If asked about topics outside astronomy or physics, politely redirect to your areas of expertise.
        
        Incorporate the context provided from the research papers when answering questions, and cite specific papers when relevant.
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
    response = chatbot.chat("What are Professor Wechsler's views on dark matter?")
    print(response["answer"])
    print("\nSources:", response["sources"]) 
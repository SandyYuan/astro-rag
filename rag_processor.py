import os
import PyPDF2
from io import BytesIO
import logging
import json
import pickle
from typing import List, Dict, Any

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, papers_dir="papers", output_dir="rag_data", provider=None):
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the LLM provider
        self.llm_provider = LLMProvider(provider=provider)
        
    def load_papers(self) -> List[Document]:
        """Load PDF papers and convert to LangChain documents."""
        all_docs = []
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(self.papers_dir, pdf_file)
                logger.info(f"Loading paper: {file_path}")
                
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata["source"] = pdf_file
                
                all_docs.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} pages from {pdf_file}")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
        
        logger.info(f"Loaded a total of {len(all_docs)} document chunks")
        return all_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents: List[Document]):
        """Create a vector store from the document chunks."""
        provider_name = self.llm_provider.provider.capitalize()
        logger.info(f"Creating vector store with {provider_name} embeddings...")
        
        # Get embeddings from the provider
        embeddings = self.llm_provider.get_embeddings()
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save the vector store
        vector_store_path = os.path.join(self.output_dir, "vector_store")
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store saved to {vector_store_path}")
        
        return vector_store
    
    def save_document_metadata(self, documents: List[Document]):
        """Save metadata about the documents for reference."""
        metadata = {
            "document_count": len(documents),
            "sources": list(set(doc.metadata.get("source", "") for doc in documents)),
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def process(self):
        """Process all papers and build the RAG system."""
        # Load papers
        documents = self.load_papers()
        if not documents:
            logger.error("No documents loaded. Make sure the papers directory contains PDF files.")
            return None
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Save metadata
        self.save_document_metadata(chunks)
        
        return vector_store

if __name__ == "__main__":
    processor = RAGProcessor()
    vector_store = processor.process() 
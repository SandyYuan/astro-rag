import os
import PyPDF2
from io import BytesIO
import logging
import json
import pickle
from typing import List, Dict, Any
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self, papers_dir="papers", output_dir="rag_data", provider=None):
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the LLM provider
        self.llm_provider = LLMProvider(provider=provider)
        
    def load_papers(self) -> List[Document]:
        """Load PDF papers and convert to LangChain documents."""
        all_docs = []
        
        # Check for checkpoint
        processed_files = self._load_checkpoint()
        
        pdf_files = [f for f in os.listdir(self.papers_dir) if f.endswith('.pdf')]
        remaining_files = [f for f in pdf_files if f not in processed_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files, {len(processed_files)} already processed")
        
        # Create a progress bar
        for pdf_file in tqdm(remaining_files, desc="Loading PDFs"):
            try:
                file_path = os.path.join(self.papers_dir, pdf_file)
                
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata["source"] = pdf_file
                
                all_docs.extend(docs)
                
                # Update checkpoint after each file
                processed_files.append(pdf_file)
                self._save_checkpoint(processed_files)
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
        
        logger.info(f"Loaded a total of {len(all_docs)} document chunks")
        return all_docs
    
    def _load_checkpoint(self) -> List[str]:
        """Load checkpoint of processed files."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
        return []
    
    def _save_checkpoint(self, processed_files: List[str]):
        """Save checkpoint of processed files."""
        try:
            with open(self.checkpoint_path, 'wb') as f:
                pickle.dump(processed_files, f)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        # Using 8000 character chunks with 15% overlap
        chunk_size = 8000
        overlap = int(chunk_size * 0.15)  # 15% overlap
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def create_vector_store(self, documents: List[Document]):
        """Create a vector store from the document chunks."""
        provider_name = self.llm_provider.provider.capitalize()
        logger.info(f"Creating vector store with {provider_name} embeddings...")
        
        # Get embeddings from the provider
        embeddings = self.llm_provider.get_embeddings()
        
        # Process embeddings in batches to handle API rate limits
        batch_size = 32
        total_chunks = len(documents)
        
        # Check for existing vector store to append to
        vector_store_path = os.path.join(self.output_dir, "vector_store")
        if os.path.exists(vector_store_path):
            logger.info(f"Loading existing vector store from {vector_store_path}")
            vector_store = FAISS.load_local(vector_store_path, embeddings)
        else:
            # Initialize with a small batch to create the store
            initial_batch = min(batch_size, total_chunks)
            logger.info(f"Creating new vector store with initial {initial_batch} documents")
            vector_store = FAISS.from_documents(documents[:initial_batch], embeddings)
            
            # Save initial vector store
            vector_store.save_local(vector_store_path)
            
            # Start from the next batch
            documents = documents[initial_batch:]
        
        # Process remaining documents in batches with progress bar
        for i in tqdm(range(0, len(documents), batch_size), desc="Creating embeddings"):
            batch_end = min(i + batch_size, len(documents))
            batch = documents[i:batch_end]
            
            try:
                # Add batch to the vector store
                vector_store.add_documents(batch)
                
                # Save after each batch to checkpoint progress
                vector_store.save_local(vector_store_path)
                
                # Slight delay to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                logger.info("Saving progress and pausing. You can resume by running the script again.")
                vector_store.save_local(vector_store_path)
                # If it's an API rate limit error, wait and retry
                if "rate limit" in str(e).lower():
                    retry_wait = 60  # Wait 60 seconds
                    logger.info(f"Rate limit hit. Waiting {retry_wait} seconds before retrying...")
                    time.sleep(retry_wait)
                    # Retry this batch
                    try:
                        vector_store.add_documents(batch)
                        vector_store.save_local(vector_store_path)
                    except Exception as retry_e:
                        logger.error(f"Retry failed: {str(retry_e)}")
                        break
        
        logger.info(f"Vector store creation completed. Final vector store saved to {vector_store_path}")
        return vector_store
    
    def save_document_metadata(self, documents: List[Document]):
        """Save metadata about the documents for reference."""
        metadata = {
            "document_count": len(documents),
            "sources": list(set(doc.metadata.get("source", "") for doc in documents)),
            "chunk_size": 8000,
            "chunk_overlap": 1200,  # 15% of 8000
            "embedding_model": self.llm_provider.provider,
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def process(self):
        """Process all papers and build the RAG system."""
        logger.info("Starting RAG processing pipeline")
        
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
        
        logger.info("RAG processing completed successfully")
        return vector_store

if __name__ == "__main__":
    processor = RAGProcessor()
    vector_store = processor.process() 
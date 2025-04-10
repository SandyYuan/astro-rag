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
        self.checkpoint_path = os.path.join(output_dir, "pdf_checkpoint.pkl") # Re-adding PDF checkpoint path
        self.vector_store_path = os.path.join(self.output_dir, "vector_store")
        os.makedirs(output_dir, exist_ok=True)
        
        self.llm_provider = LLMProvider(provider=provider)

    # --- PDF Loading and Checkpointing (Re-added) --- 
    def _load_pdf_checkpoint(self) -> List[str]:
        """Load checkpoint of processed PDF files."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'rb') as f:
                    logger.info(f"Attempting to load PDF checkpoint from {self.checkpoint_path}") # Added info log
                    data = pickle.load(f)
                    logger.info(f"Successfully loaded {len(data)} entries from PDF checkpoint.") # Added success log
                    return data
            except Exception as e:
                # Log the full traceback for detailed debugging
                logger.error(f"Failed to load PDF checkpoint: {str(e)}", exc_info=True) 
                logger.warning("Assuming no files processed due to checkpoint load failure.")
        else:
            logger.info("PDF checkpoint file not found. Assuming no files processed.") # Added info log for non-existence
        return []

    def _save_pdf_checkpoint(self, processed_files: List[str]):
        """Save checkpoint of processed PDF files."""
        try:
            with open(self.checkpoint_path, 'wb') as f:
                # Ensure we save a unique list
                pickle.dump(list(set(processed_files)), f)
        except Exception as e:
            logger.error(f"Error saving PDF checkpoint: {str(e)}")

    def _load_documents_from_files(self, files_to_load: List[str], desc: str) -> List[Document]:
        """Helper to load documents from a list of PDF files with progress."""
        loaded_docs = []
        successfully_loaded_files = [] # Track successfully loaded files in this batch
        for pdf_file in tqdm(files_to_load, desc=desc):
            try:
                file_path = os.path.join(self.papers_dir, pdf_file)
                if not os.path.exists(file_path):
                     logger.warning(f"PDF file not found, skipping: {pdf_file}")
                     continue
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = pdf_file
                loaded_docs.extend(docs)
                successfully_loaded_files.append(pdf_file) # Mark as successfully loaded
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
        # Return both the documents and the list of files actually loaded
        return loaded_docs, successfully_loaded_files 

    # --- Splitting ---    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        if not documents:
             return []
             
        chunk_size = 8000
        overlap = int(chunk_size * 0.15)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        
        logger.info(f"Splitting {len(documents)} document pages into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks

    # --- Vector Store Creation / Update / Resume --- 
    def create_or_update_vector_store(self, all_expected_chunks: List[Document]):
        """Create or update a FAISS vector store, resuming if necessary."""
        provider_name = self.llm_provider.provider.capitalize()
        logger.info(f"Starting vector store processing with {provider_name} embeddings.")

        if not all_expected_chunks:
            logger.warning("No document chunks provided to create or update the vector store.")
            # Try loading existing store anyway, maybe it just needs loading
            if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
                 try:
                      embeddings = self.llm_provider.get_embeddings()
                      vector_store = FAISS.load_local(self.vector_store_path, embeddings, allow_dangerous_deserialization=True)
                      logger.info(f"Loaded existing vector store with {vector_store.index.ntotal} entries. No new chunks to add.")
                      return vector_store
                 except Exception as e:
                      logger.error(f"Failed to load existing vector store even with no new chunks: {e}")
                      return None
            else:
                 logger.error("No chunks provided and no existing store found.")
                 return None

        embeddings = self.llm_provider.get_embeddings()
        vector_store = None
        start_index = 0

        # Try loading existing store to find resume point
        if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            try:
                logger.info(f"Loading existing vector store from {self.vector_store_path} to check status.")
                vector_store = FAISS.load_local(self.vector_store_path, embeddings, allow_dangerous_deserialization=True)
                start_index = vector_store.index.ntotal
                logger.info(f"Existing store loaded with {start_index} entries.")
                
                if start_index >= len(all_expected_chunks):
                     logger.info("Existing vector store contains all expected chunks. Processing complete.")
                     return vector_store
                else:
                     logger.info(f"Resuming embedding process from chunk index {start_index}. Need to process {len(all_expected_chunks) - start_index} more chunks.")

            except Exception as e:
                logger.warning(f"Error loading existing vector store: {str(e)}. Will attempt to recreate from scratch.")
                vector_store = None # Force recreation
                start_index = 0

        # Determine chunks to process for this run
        chunks_to_process = all_expected_chunks[start_index:]
        if not chunks_to_process:
             logger.info("No chunks remaining to process.")
             return vector_store

        logger.info(f"Processing {len(chunks_to_process)} chunks (starting from overall index {start_index})...")
        batch_size = 32
        
        # If vector store doesn't exist (or failed to load), create it with the first batch
        if vector_store is None:
             logger.info("Creating new vector store.")
             initial_batch_docs = chunks_to_process[:batch_size]
             if not initial_batch_docs:
                  logger.error("Cannot create vector store: No documents available for initial batch.")
                  return None
             try:
                  logger.info(f"Creating store with first {len(initial_batch_docs)} chunks.")
                  vector_store = FAISS.from_documents(initial_batch_docs, embeddings)
                  vector_store.save_local(self.vector_store_path)
                  logger.info(f"Initial vector store created and saved. Processed up to index {start_index + len(initial_batch_docs) - 1}.")
                  # Adjust list for the main loop
                  chunks_to_process = chunks_to_process[batch_size:]
                  start_index += len(initial_batch_docs) # Update effective start for tqdm description
             except Exception as e:
                  logger.error(f"Failed to create initial vector store: {str(e)}")
                  return None
        
        if not chunks_to_process:
             logger.info("Initial batch processing completed the vector store.")
             return vector_store

        # Process remaining chunks in batches
        desc = f"Adding embeddings (Index {start_index}-{len(all_expected_chunks)-1})"
        for i in tqdm(range(0, len(chunks_to_process), batch_size), desc=desc):
            batch_docs = chunks_to_process[i:min(i + batch_size, len(chunks_to_process))]
            if not batch_docs: continue
            current_chunk_index = start_index + i # For logging

            try:
                vector_store.add_documents(batch_docs)
                vector_store.save_local(self.vector_store_path)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing batch starting at overall index ~{current_chunk_index}: {str(e)}")
                logger.info("Saving progress before potential pause/retry.")
                vector_store.save_local(self.vector_store_path) # Ensure latest state is saved
                if "rate limit" in str(e).lower():
                    retry_wait = 60
                    logger.info(f"Rate limit hit. Waiting {retry_wait} seconds...")
                    time.sleep(retry_wait)
                    try:
                        logger.info(f"Retrying batch starting at index ~{current_chunk_index}...")
                        vector_store.add_documents(batch_docs)
                        vector_store.save_local(self.vector_store_path)
                        logger.info("Retry successful.")
                    except Exception as retry_e:
                        logger.error(f"Retry failed for batch at index ~{current_chunk_index}: {str(retry_e)}. Stopping.")
                        return None # Return None on retry failure
                else:
                     logger.error("Non-rate-limit error. Stopping embedding process.")
                     return None # Return None on non-rate-limit error

        final_count = vector_store.index.ntotal if vector_store else 0
        logger.info(f"Vector store processing completed. Final store contains {final_count} entries. Saved to {self.vector_store_path}")
        return vector_store
    
    # --- Main Process (Reinstating PDF Checkpoint Logic) --- 
    def process(self):
        """Load PDFs with checkpointing, split docs, create/update vector store."""
        logger.info("Starting RAG processing pipeline...")
        
        # 1. Check for new PDF files using checkpoint
        processed_pdf_files = self._load_pdf_checkpoint()
        try:
            pdf_files_in_dir = [f for f in os.listdir(self.papers_dir) if f.endswith('.pdf')]
        except FileNotFoundError:
             logger.error(f"Papers directory not found: {self.papers_dir}. Cannot process.")
             return None
             
        new_pdf_files = [f for f in pdf_files_in_dir if f not in processed_pdf_files]
        logger.info(f"Found {len(pdf_files_in_dir)} total PDF files. {len(processed_pdf_files)} previously processed according to checkpoint.")

        # 2. Load and process ONLY new PDFs, update checkpoint
        successfully_processed_new_files = []
        if new_pdf_files:
             logger.info(f"Loading {len(new_pdf_files)} new PDF files...")
             # _load_documents_from_files now returns docs and successfully loaded file names
             new_documents, successfully_processed_new_files = self._load_documents_from_files(new_pdf_files, desc="Loading New PDFs")
             
             if successfully_processed_new_files:
                 logger.info(f"Successfully loaded {len(successfully_processed_new_files)} new files.")
                 # Update the main checkpoint with successfully processed new files
                 current_checkpoint = self._load_pdf_checkpoint()
                 current_checkpoint.extend(successfully_processed_new_files)
                 self._save_pdf_checkpoint(current_checkpoint) # Saves unique list internally
                 logger.info(f"PDF checkpoint updated.")
             else:
                  logger.warning("Loading of new PDFs failed or yielded no documents, checkpoint not updated.")
        else:
             logger.info("No new PDF files to process based on checkpoint.")
        
        # 3. Determine the full list of PDFs that should be in the vector store (from the updated checkpoint)
        all_processed_pdf_files = self._load_pdf_checkpoint()
        if not all_processed_pdf_files:
             logger.error("No processed PDFs found in checkpoint. Cannot build or verify vector store.")
             return None
             
        # 4. Load documents for ALL files in the checkpoint for vector store processing
        logger.info(f"Loading documents for all {len(all_processed_pdf_files)} PDFs listed in checkpoint to ensure vector store completeness...")
        all_documents = self._load_documents_from_files(all_processed_pdf_files, desc="Loading All PDFs for VS")
        # Note: _load_documents_from_files returns a tuple now, we only need the docs here
        all_documents = all_documents[0] 
        
        if not all_documents:
             logger.error("Failed to load documents required by checkpoint. Cannot proceed.")
             return None

        # 5. Split ALL loaded documents into the total expected chunks
        all_expected_chunks = self.split_documents(all_documents)
        if not all_expected_chunks:
             logger.error("Splitting documents resulted in no chunks. Cannot proceed.")
             return None
        
        # 6. Create or update the vector store with ALL expected chunks (handles resume internally)
        vector_store = self.create_or_update_vector_store(all_expected_chunks)
        
        if vector_store is None:
            logger.error("RAG processing failed: Could not load or create a valid vector store.")
            return None

        logger.info("RAG processing pipeline finished successfully.")
        return vector_store

if __name__ == "__main__":
    processor = RAGProcessor()
    vector_store = processor.process() 
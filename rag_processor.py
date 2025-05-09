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
    def __init__(self, papers_dirs=["papers", "papers_np"], output_dir="rag_data", provider=None):
        self.papers_dirs = papers_dirs if isinstance(papers_dirs, list) else [papers_dirs]
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, "pdf_checkpoint.pkl")
        self.chunk_checkpoint_path = os.path.join(output_dir, "chunk_checkpoint.pkl")  # New checkpoint for chunks
        self.vector_store_path = os.path.join(self.output_dir, "index_all")
        os.makedirs(output_dir, exist_ok=True)
        
        self.llm_provider = LLMProvider(provider=provider)

    # --- PDF Loading and Checkpointing (Re-added) --- 
    def _load_pdf_checkpoint(self) -> List[str]:
        """Load checkpoint of processed PDF files."""
        logger.info(f"Looking for checkpoint file at: {os.path.abspath(self.checkpoint_path)}")
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
            logger.info(f"PDF checkpoint file not found at {os.path.abspath(self.checkpoint_path)}. Assuming no files processed.") # Added info log for non-existence
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
        
        # Group files by their directory
        files_by_dir = {}
        for pdf_file in files_to_load:
            found = False
            for papers_dir in self.papers_dirs:
                if os.path.exists(os.path.join(papers_dir, pdf_file)):
                    if papers_dir not in files_by_dir:
                        files_by_dir[papers_dir] = []
                    files_by_dir[papers_dir].append(pdf_file)
                    found = True
                    break
            if not found:
                logger.warning(f"PDF file not found in any directory: {pdf_file}")
        
        # Process files from each directory
        for papers_dir, dir_files in files_by_dir.items():
            logger.info(f"Processing files from directory: {papers_dir}")
            for pdf_file in tqdm(dir_files, desc=f"Loading PDFs from {papers_dir}"):
                try:
                    file_path = os.path.join(papers_dir, pdf_file)
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = f"{papers_dir}/{pdf_file}"  # Include directory in source
                    loaded_docs.extend(docs)
                    successfully_loaded_files.append(pdf_file)
                except Exception as e:
                    logger.error(f"Error loading {pdf_file}: {str(e)}")
        
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

    def _load_chunk_checkpoint(self) -> set:
        """Load checkpoint of processed chunks."""
        if os.path.exists(self.chunk_checkpoint_path):
            try:
                with open(self.chunk_checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load chunk checkpoint: {str(e)}")
        return set()

    def _save_chunk_checkpoint(self, processed_chunks: set):
        """Save checkpoint of processed chunks."""
        try:
            with open(self.chunk_checkpoint_path, 'wb') as f:
                pickle.dump(processed_chunks, f)
        except Exception as e:
            logger.error(f"Error saving chunk checkpoint: {str(e)}")

    # --- Vector Store Creation / Update / Resume --- 
    def create_or_update_vector_store(self, all_expected_chunks: List[Document]):
        """Create or update a FAISS vector store, resuming if necessary."""
        provider_name = "Google"  # Since we're using Google's Gemini models
        logger.info(f"Starting vector store processing with {provider_name} embeddings.")

        # Load chunk checkpoint
        processed_chunks = self._load_chunk_checkpoint()
        logger.info(f"Found {len(processed_chunks)} previously processed chunks in checkpoint")

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
            except Exception as e:
                logger.warning(f"Error loading existing vector store: {str(e)}. Will attempt to recreate from scratch.")
                vector_store = None # Force recreation
                start_index = 0

        # Filter out already processed chunks
        chunks_to_process = []
        for i, chunk in enumerate(all_expected_chunks):
            chunk_id = f"{chunk.metadata.get('source', 'unknown')}_{i}"
            if chunk_id not in processed_chunks:
                chunks_to_process.append((i, chunk))

        if not chunks_to_process:
            logger.info("All chunks have been processed according to checkpoint.")
            return vector_store

        logger.info(f"Processing {len(chunks_to_process)} remaining chunks...")
        batch_size = 8  # Reduced from 16 to 8 to avoid timeouts
        
        # If vector store doesn't exist (or failed to load), create it with the first batch
        if vector_store is None:
             logger.info("Creating new vector store.")
             initial_batch = chunks_to_process[:batch_size]
             if not initial_batch:
                  logger.error("Cannot create vector store: No documents available for initial batch.")
                  return None
             try:
                  initial_docs = [doc for _, doc in initial_batch]
                  logger.info(f"Creating store with first {len(initial_docs)} chunks.")
                  vector_store = FAISS.from_documents(initial_docs, embeddings)
                  vector_store.save_local(self.vector_store_path)
                  
                  # Update checkpoint with successfully processed chunks
                  for i, _ in initial_batch:
                      chunk_id = f"{all_expected_chunks[i].metadata.get('source', 'unknown')}_{i}"
                      processed_chunks.add(chunk_id)
                  self._save_chunk_checkpoint(processed_chunks)
                  
                  logger.info(f"Initial vector store created and saved. Processed up to index {len(initial_batch) - 1}.")
                  chunks_to_process = chunks_to_process[batch_size:]
             except Exception as e:
                  logger.error(f"Failed to create initial vector store: {str(e)}")
                  return None
        
        if not chunks_to_process:
             logger.info("Initial batch processing completed the vector store.")
             return vector_store

        # Process remaining chunks in batches
        for i in tqdm(range(0, len(chunks_to_process), batch_size), desc="Adding embeddings"):
            batch = chunks_to_process[i:min(i + batch_size, len(chunks_to_process))]
            if not batch: continue
            
            batch_docs = [doc for _, doc in batch]
            batch_indices = [idx for idx, _ in batch]
            current_chunk_index = batch_indices[0]

            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    vector_store.add_documents(batch_docs)
                    vector_store.save_local(self.vector_store_path)
                    
                    # Update checkpoint with successfully processed chunks
                    for idx in batch_indices:
                        chunk_id = f"{all_expected_chunks[idx].metadata.get('source', 'unknown')}_{idx}"
                        processed_chunks.add(chunk_id)
                    self._save_chunk_checkpoint(processed_chunks)
                    
                    time.sleep(3)  # Increased delay between batches
                    break  # Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg:
                        retry_wait = 60 * retry_count  # Progressive backoff
                        logger.info(f"Rate limit hit. Attempt {retry_count}/{max_retries}. Waiting {retry_wait} seconds...")
                        time.sleep(retry_wait)
                        continue
                    elif "deadline exceeded" in error_msg or "504" in error_msg:
                        retry_wait = 30 * retry_count  # Shorter wait for timeouts
                        logger.info(f"Deadline exceeded. Attempt {retry_count}/{max_retries}. Waiting {retry_wait} seconds...")
                        time.sleep(retry_wait)
                        # Reduce batch size for next attempt
                        if len(batch_docs) > 4:
                            batch_docs = batch_docs[:len(batch_docs)//2]
                            batch_indices = batch_indices[:len(batch_indices)//2]
                            logger.info(f"Reduced batch size to {len(batch_docs)} for next attempt")
                        continue
                    else:
                        logger.error(f"Error processing batch starting at overall index ~{current_chunk_index}: {str(e)}")
                        logger.info("Saving progress before potential pause/retry.")
                        vector_store.save_local(self.vector_store_path)
                        return None  # Return None on non-rate-limit error
            
            if retry_count == max_retries:
                logger.error(f"Failed to process batch after {max_retries} retries. Stopping.")
                return None

        final_count = vector_store.index.ntotal if vector_store else 0
        logger.info(f"Vector store processing completed. Final store contains {final_count} entries. Saved to {self.vector_store_path}")
        return vector_store
    
    # --- Main Process (Reinstating PDF Checkpoint Logic) --- 
    def process(self):
        """Load PDFs with checkpointing, split docs, create/update vector store."""
        logger.info("Starting RAG processing pipeline...")
        
        # 1. Check for new PDF files using checkpoint
        processed_pdf_files = self._load_pdf_checkpoint()
        
        # Get all PDF files from all directories
        pdf_files_in_dirs = []
        for papers_dir in self.papers_dirs:
            try:
                dir_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]
                pdf_files_in_dirs.extend(dir_files)
            except FileNotFoundError:
                logger.error(f"Papers directory not found: {papers_dir}. Skipping.")
                continue
        
        new_pdf_files = [f for f in pdf_files_in_dirs if f not in processed_pdf_files]
        logger.info(f"Found {len(pdf_files_in_dirs)} total PDF files. {len(processed_pdf_files)} previously processed according to checkpoint.")

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
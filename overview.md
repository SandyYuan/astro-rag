## Project Overview

Professor-specific chatbot that uses Retrieval-Augmented Generation (RAG) over research papers. It collects papers (arXiv), builds a vector index (FAISS) with Gemini embeddings, and serves a FastAPI web UI for chat. Persona and tone are controlled by a system prompt and an optional summary file.

### Key Capabilities
- Ingestion: arXiv paper downloader with metadata capture
- Processing: robust, resumable pipeline to split, embed, and index documents
- Retrieval: FAISS + MMR retriever; sources returned with answers
- Generation: Gemini LLM; persona-driven responses
- Serving: FastAPI app with a simple browser UI

## Repository Structure
- `app.py`: FastAPI server and web UI bootstrap (serves `templates/index_modern.html`). Initializes the chatbot.
- `chatbot.py`: Core chat workflow
  - Loads FAISS index and builds a retriever (`mmr`, `k=5`, `fetch_k=10`, `lambda_mult=0.7`)
  - Constructs a custom QA chain (LangChain `load_qa_chain` with `chain_type="stuff"`)
  - Maintains minimal chat history and uses a persona/system prompt
  - Accepts an optional summary file (`rag_data/prof_summary.txt`) to enrich persona
- `rag_processor.py`: Index building
  - Loads PDFs via `PyPDFLoader`
  - Splits with `RecursiveCharacterTextSplitter` (chunk_size=8000, overlap≈15%)
  - Embeds with Gemini `text-embedding-004` (via `langchain-google-genai`)
  - Stores vectors in FAISS at `rag_data/index_all`
  - Checkpointing for PDFs (`rag_data/pdf_checkpoint.pkl`) and chunk batches (`rag_data/chunk_checkpoint.pkl`) with resume/backoff
- `paper_collector.py`: arXiv search + download utility for a target author
  - Saves PDFs to `papers/` and `.txt` metadata/abstracts alongside
  - Optionally collects non-primary-author papers to `papers/papers-np` (this repo uses `papers_np/`)
- `llm_provider.py`: Gemini client wrapper and LangChain-compatible LLM/Embeddings factories
- `templates/`, `static/`: Web assets (template written on first request)
- `rag_data/`: Vector store and auxiliary files (created at runtime)
- `papers/`, `papers_np/`: Paper PDFs and `.txt` companions
- `README.md`: Setup and usage
- `graph.md`: Plan for optional Neo4j GraphRAG integration (future/optional)

## Data Flow
1. Collect papers: `paper_collector.py` downloads PDFs and writes metadata `.txt`
2. Build index: `rag_processor.py` loads PDFs, splits to chunks, embeds with Gemini, and updates FAISS
3. Serve chat: `app.py` initializes `AstronomyChatbot` which loads FAISS and serves a chat UI

## Environment & Dependencies
- Python deps in `requirements.txt` (LangChain, FAISS, FastAPI, Google Generative AI, PyPDF, arxiv)
- Required env var: `GOOGLE_API_KEY`
  - Optional legacy: `LLM_PROVIDER=google` (kept for compatibility)

### Conda environment activation (mcp)
This is necessary to run anything. Make sure to run this before testing the code. 
- bash
  ```bash
eval "$(conda shell.bash hook)" && conda activate mcp
  ```
- If the hook is unavailable, source conda explicitly (adjust path if needed):
  ```bash
source /Users/sandyyuan/opt/anaconda3/etc/profile.d/conda.sh && conda activate mcp
  ```
- Without activation (one-off):
  ```bash
conda run -n mcp python -V
  ```

Place a `.env` file in repo root, e.g.:
```
GOOGLE_API_KEY=your_google_api_key
```

## Quick Start (local)
1) Install
```
pip install -r requirements.txt
```

2) Set API key
```
echo "GOOGLE_API_KEY=your_google_api_key" > .env
```

3) (Optional) Collect papers for a different author
```
python paper_collector.py
```

4) Build or update the FAISS index (resumable)
```
python rag_processor.py
```

5) Run the web app
```
python app.py
# Visit http://localhost:8000
```

## Defaults & Important Paths
- Vector store path loaded by the app: `rag_data/index_all`
  - `app.initialize_chatbot()` points `vector_store_path` to `rag_data/index_all`
- Summary/persona file (optional): `rag_data/prof_summary.txt`
- Checkpoints: `rag_data/pdf_checkpoint.pkl`, `rag_data/chunk_checkpoint.pkl`

## How To Extend or Modify
- Change the persona: edit `AstronomyChatbot.get_system_prompt()` in `chatbot.py` and/or provide `rag_data/prof_summary.txt`
- Tweak retrieval: adjust `k`, `fetch_k`, `lambda_mult` in `chatbot.py` retriever setup
- Adjust chunking: change `chunk_size` or `chunk_overlap` in `rag_processor.py`
- Switch embedding/LLM models: update defaults in `llm_provider.py`

## Common Issues & Tips
- Missing index errors: run `python rag_processor.py` to create `rag_data/index_all`
- API errors/limits: ensure `GOOGLE_API_KEY` is valid; the processor backs off and resumes on rate/timeouts
- Sources not shown: ensure `Document.metadata["source"]` is set (handled during PDF load)
- Path consistency: the app expects `rag_data/index_all` (ensure `rag_processor.py` writes there)

## Optional: Knowledge-Graph RAG
See `graph.md` for a minimal, side-by-side Neo4j GraphRAG plan. It adds a `graph_rag/` module and a small `RAG_MODE` switch in `chatbot.py` without changing the API.



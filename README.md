# Astronomy RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot focused on astronomy topics. It uses Google's Gemini models for text generation and embeddings.

## Features

- Web-based chat interface
- RAG system for knowledge retrieval
- Conversation context memory
- Resilient implementation with fallback mechanisms
- Deployable to Google Cloud Run or similar platforms

## Fallback Implementation

This application includes a lightweight fallback implementation of the Google Generative AI client. This works even in environments where the official Google Generative AI client can't be installed, such as in some cloud environments with restricted permissions.

The fallback implementation (`fallback_genai.py`) provides:

- A simple HTTP-based client for Gemini APIs
- Support for text generation with the Gemini models
- Support for embeddings with Google's embedding models
- Automatic fallback when the official client can't be loaded

## Deployment

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open http://localhost:8000 in your browser

### Docker Deployment

1. Run the build script: `./build-image.sh`
2. Follow the instructions to push the image to Docker Hub
3. Deploy to Google Cloud Run or similar platforms

### Cloud Run Deployment

```bash
# Push to Docker Hub
docker login
docker push <username>/astro-rag-repo:latest

# Deploy to Cloud Run
gcloud run deploy --image <username>/astro-rag-repo:latest --platform managed --allow-unauthenticated
```

## API Key Setup

This application requires a Google Gemini API key to function. Users will need to provide their API key through the web interface. The API key is only stored in memory during the session and is not persisted.

## Architecture

- `app.py`: FastAPI web application
- `chatbot.py`: RAG implementation with conversation memory
- `llm_provider.py`: Provider for LLM services with resilient loading
- `fallback_genai.py`: Fallback implementation of Google Generative AI client

## Project Overview

This project builds a conversational AI system that:
1. Collects research papers by a specific professor (currently using Risa Wechsler as an example)
2. Processes these papers into a searchable knowledge base
3. Uses RAG technology with Gemini to provide informed responses in the style of the professor
4. Hosts the chatbot through a web interface

## Architecture

The chatbot combines multiple AI techniques to create natural, informative, and contextually-aware conversations:

### Dual-Augmentation Approach

1. **RAG (Retrieval-Augmented Generation)**
   - Indexes professor's research papers in a FAISS vector database
   - Uses semantic search to retrieve relevant document fragments for each query
   - Embeds documents using Google's text-embedding-004 model
   - Employs Maximum Marginal Relevance (MMR) for diverse, relevant results
   - Retrieved documents are combined with the prompt before being sent to the LLM

2. **CAG (Context-Augmented Generation)**
   - Incorporates a pre-written summary file (`prof_summary.txt`) with biographical information
   - Provides personality, style, and general stance information about the professor
   - Helps the model respond to questions about career, opinions, and non-research topics
   - This static context is combined with the dynamic RAG results

### Conversation Context Management

The system maintains conversation history through an innovative dual-context approach:

1. **Document Retrieval Context**
   - For follow-up questions, previous queries are included in the retrieval query
   - Example: If a user asks "What is dark matter?" followed by "Why is it important?", the retrieval query becomes "Context: What is dark matter? Question: Why is it important?"
   - This helps the system retrieve documents relevant to the entire conversation flow

2. **Response Generation Context**
   - Stores the last 3 conversation exchanges (question-answer pairs)
   - Includes this conversational history in the prompt to the LLM
   - Explicitly instructs the model to maintain continuity with previous exchanges
   - Preserves the "thread" of conversation across multiple turns

3. **Single-Stage RAG Implementation**
   - Uses a custom document QA chain with direct document processing
   - Manually retrieves documents using context-enhanced queries
   - Combines system instructions, conversation history, and retrieved documents in a carefully crafted prompt

This architecture ensures the chatbot can handle follow-up questions naturally, maintain professor-specific knowledge, and provide responses that feel like a cohesive conversation rather than isolated Q&A pairs.

## Components

- `paper_collector.py`: Tool to search and download papers by a target professor
- `rag_processor.py`: Processes papers and creates the vector database for RAG using Gemini embeddings
- `chatbot.py`: Core chatbot implementation using Gemini and RAG
- `app.py`: Web application to host the chatbot
- `requirements.txt`: Dependencies for the project

## Setup Instructions

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key in a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Configure the target professor (defaults to Risa Wechsler as an example):
   ```python
   # In paper_collector.py, modify:
   collector = PaperCollector(author_name="Professor Name")
   ```

5. Run the paper collector to gather research content:
   ```
   python paper_collector.py
   ```

6. Process the papers to build the RAG system:
   ```
   python rag_processor.py
   ```

7. Start the web application:
   ```
   python app.py
   ```

8. Access the chatbot at `http://localhost:8000`

## Usage

Once the web application is running, you can interact with the chatbot through the web interface:

1. Type your question in the input field
2. Press "Send" or hit Enter
3. The chatbot will respond based on the professor's research papers
4. Sources used to generate the response will be displayed below each answer

## Customization

### Adjusting the Chatbot Personality

You can modify the system prompt in `chatbot.py` to refine how the chatbot emulates the professor.

### Adding More Papers

To expand the knowledge base, run the paper collector again with higher `max_papers` value:

```python
collector = PaperCollector(author_name="Professor Name")
papers = collector.collect_papers(max_papers=50)
```

Then reprocess the papers to update the vector database.

## Dependencies

- LangChain for RAG implementation
- Google Generative AI for embeddings and language model
- FAISS for vector storage
- FastAPI for web application
- Scholarly for paper collection

## Notes

- The system requires a Google API key with access to Gemini models
- Paper collection may be limited by API rate limits
- For educational and research purposes only 
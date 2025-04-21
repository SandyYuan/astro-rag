# Academic Research Assistant: Professor-Specific Chatbot

A specialized chatbot system that emulates academic professors, using their research papers as a knowledge base with Retrieval-Augmented Generation (RAG) powered by Google's Gemini AI models.

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
   git clone https://github.com/SandyYuan/astro-rag.git
   cd astro-rag
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key in a `.env` file:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. Start the web application:
   ```
   python app.py
   ```

5. Access the chatbot at `http://localhost:8000` (This version does not include the in-context summary file. You can ask me for it.)

If you would like to run your own literature database or emulate a different professor:


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
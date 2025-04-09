# Academic Research Assistant: Professor-Specific Chatbot

A specialized chatbot system that emulates academic professors, using their research papers as a knowledge base with Retrieval-Augmented Generation (RAG) powered by Google's Gemini AI models.

## Project Overview

This project builds a conversational AI system that:
1. Collects research papers by a specific professor (currently using Risa Wechsler as an example)
2. Processes these papers into a searchable knowledge base
3. Uses RAG technology with Gemini to provide informed responses in the style of the professor
4. Hosts the chatbot through a web interface

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
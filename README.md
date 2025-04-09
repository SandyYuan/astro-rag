# Astronomy Chatbot: Professor Risa Wechsler

A specialized chatbot that emulates Professor Risa Wechsler, using her research papers as a knowledge base with Retrieval-Augmented Generation (RAG).

## Project Overview

This project builds a conversational AI system that:
1. Collects research papers by Professor Risa Wechsler
2. Processes these papers into a searchable knowledge base
3. Uses RAG technology to provide informed responses in the style of Professor Wechsler
4. Hosts the chatbot through a web interface

## Components

- `paper_collector.py`: Tool to search and download papers by Professor Wechsler
- `rag_processor.py`: Processes papers and creates the vector database for RAG
- `chatbot.py`: Core chatbot implementation using RAG
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

3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the paper collector to gather research content:
   ```
   python paper_collector.py
   ```

5. Process the papers to build the RAG system:
   ```
   python rag_processor.py
   ```

6. Start the web application:
   ```
   python app.py
   ```

7. Access the chatbot at `http://localhost:8000`

## Usage

Once the web application is running, you can interact with the chatbot through the web interface:

1. Type your question in the input field
2. Press "Send" or hit Enter
3. The chatbot will respond based on Professor Wechsler's research papers
4. Sources used to generate the response will be displayed below each answer

## Customization

### Adjusting the Chatbot Personality

You can modify the system prompt in `chatbot.py` to refine how the chatbot emulates Professor Wechsler.

### Adding More Papers

To expand the knowledge base, run the paper collector again with higher `max_papers` value:

```python
collector = PaperCollector()
papers = collector.collect_papers(max_papers=50)
```

Then reprocess the papers to update the vector database.

## Dependencies

- LangChain for RAG implementation
- OpenAI for embeddings and language model
- FAISS for vector storage
- FastAPI for web application
- Scholarly for paper collection

## Notes

- The system requires an OpenAI API key with access to embedding and language models
- Paper collection may be limited by API rate limits
- For educational and research purposes only 
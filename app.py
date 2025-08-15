import os
import logging
import re  # Added for regex pattern matching
import json
import asyncio
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import dotenv

from chatbot import AstronomyChatbot
from llm_provider import LLMProvider

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(title="Astronomy Chatbot")

# Create the directories for templates and static files
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize the chatbot
chatbot = None

class ChatRequest(BaseModel):
    message: str

def initialize_chatbot():
    """Initialize the chatbot with the correct vector store path."""
    global chatbot
    try:
        # Initialize with the new vector store path
        chatbot = AstronomyChatbot(
            vector_store_path="rag_data/index_all",
        )
        logger.info("Chatbot initialized successfully with new vector store")
        # Log retrieval mode for visibility
        try:
            logger.info(f"Retrieval mode: {getattr(chatbot, 'retrieval_mode', 'faiss')}")
        except Exception:
            # Keep initialization strict; this log should not interfere with startup
            pass
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot when the server starts."""
    initialize_chatbot()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page directly from the curated template."""
    return templates.TemplateResponse("index_modern.html", {"request": request})

    # Dead-code fallback removed by design to fail fast if template missing.
    # If you prefer a soft fallback, restore the block below.
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mini-Risa Chatbot</title>
        <!-- Add Marked.js library from CDN -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .model-info {
                text-align: center;
                color: #666;
                margin-bottom: 20px;
                font-style: italic;
            }
            .chat-container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            #chat-messages {
                height: 75vh;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .message {
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background-color: #e3f2fd;
                text-align: left;
                border-radius: 18px 18px 0 18px;
                margin-left: auto;
                margin-right: 0;
            }
            .bot-message {
                background-color: #f1f1f1;
                border-radius: 18px 18px 18px 0;
                margin-left: 0;
                margin-right: auto;
                text-align: left;
            }
            .sources {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }
            .chat-input-container {
                display: flex;
            }
            #user-input {
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 4px;
                margin-left: 10px;
                cursor: pointer;
            }
            button:hover {
                background-color: #3367d6;
            }
            .loading {
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>Mini-Risa Chatbot</h1>
        
        <div class="chat-container">
            <div id="chat-messages">
                <div class="message bot-message">
                    Hello! I'm a chatbot designed to emulate Professor Risa Wechsler, an astrophysicist and cosmologist. 
                    How can I help with questions about astronomy, cosmology, dark matter, or related topics?
                </div>
            </div>
            
            <div class="chat-input-container">
                <input 
                    type="text" 
                    id="user-input" 
                    placeholder="Type your question here..." 
                    autocomplete="off"
                >
                <button id="send-button">Send</button>
            </div>
        </div>
        
        <script>
            const messagesContainer = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            console.log('UI elements initialized:', {
                messagesContainer: !!messagesContainer,
                userInput: !!userInput,
                sendButton: !!sendButton
            });
            
            // Function to add a message to the chat
            function addMessage(content, isUser, sources = []) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
                
                // Render content: Use Markdown for bot, plain text for user
                if (isUser) {
                    messageDiv.textContent = content;
                } else {
                    if (typeof marked !== 'undefined') {
                        // Use marked.parse() which handles sanitization by default
                        messageDiv.innerHTML = marked.parse(content);
                    } else {
                        // Fallback if marked.js fails to load
                        console.warn("Marked.js not loaded. Falling back to newline replacement.");
                        const newlineRegex = new RegExp('\\n', 'g');
                        messageDiv.innerHTML = content.replace(newlineRegex, '<br>');
                    }
                }
                
                // Add sources if available
                if (sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.textContent = 'Sources: ' + sources.join(', ');
                    messageDiv.appendChild(sourcesDiv);
                }
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            // Function to send message to the server
            async function sendMessage() {
                console.log('sendMessage function called');
                const message = userInput.value.trim();
                console.log('Message content:', message);
                if (!message) {
                    console.log('Message is empty, returning');
                    return;
                }
                
                // Add user message to chat
                addMessage(message, true);
                
                // Clear input
                userInput.value = '';
                
                // Add loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message bot-message loading';
                loadingDiv.textContent = 'Thinking...';
                messagesContainer.appendChild(loadingDiv);
                
                try {
                    console.log('Sending request to server...');
                    // Send to server with absolute URL
                    const chatEndpoint = new URL('/chat', window.location.href).href;
                    console.log('Using endpoint:', chatEndpoint);
                    
                    const response = await fetch(chatEndpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message }),
                    });
                    
                    console.log('Response status:', response.status);
                    if (!response.ok) {
                        throw new Error('Failed to get response: ' + response.status);
                    }
                    
                    const data = await response.json();
                    
                    // Remove loading indicator
                    messagesContainer.removeChild(loadingDiv);
                    
                    // Post-process the answer to remove unwanted phrases
                    let answer = data.answer;
                    const phrasesToRemove = [
                        "Based on the provided text, ",
                        "Based on the provided texts, ",
                        "According to the documents, ",
                        "According to the text, ",
                        "The context suggests that ",
                        "The provided context indicates that ",
                        "From the text provided, ",
                        "In the provided text, ",
                        "Based on the context, ",
                        "The text indicates that ",
                        "From the documents provided, ",
                        "According to the provided information, ",
                        "The information provided suggests that ",
                        "Based on the information given, "
                    ];
                    
                    // First check for phrases at the beginning of the response
                    for (const phrase of phrasesToRemove) {
                        if (answer.startsWith(phrase)) {
                            answer = answer.substring(phrase.length);
                            break;
                        }
                    }
                    
                    // Then check for these phrases anywhere in the text
                    for (const phrase of phrasesToRemove) {
                        // Create a regex pattern that's case-insensitive
                        const pattern = new RegExp('\\s*' + phrase.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'), 'gi');
                        answer = answer.replace(pattern, ' ');
                    }
                    
                    // Update the answer
                    data.answer = answer.trim();
                    
                    // Add bot response
                    addMessage(data.answer, false, data.sources);
                } catch (error) {
                    // Remove loading indicator
                    messagesContainer.removeChild(loadingDiv);
                    
                    // Show error message
                    addMessage('Sorry, there was an error processing your request. Please try again.', false);
                    console.error('Error:', error);
                }
            }
            
            // Initialize Marked.js options (ensure script is loaded before this)
            if (typeof marked !== 'undefined') {
                marked.setOptions({
                    breaks: true, // Treat single newlines as <br>
                    gfm: true      // Enable GitHub Flavored Markdown
                });
            }
            
            // Event listeners
            console.log('Adding event listeners');
            
            sendButton.addEventListener('click', function() {
                console.log('Send button clicked');
                sendMessage();
            });
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    console.log('Enter key pressed');
                    sendMessage();
                }
            });
            
            console.log('Event listeners added successfully');
        </script>
    </body>
    </html>
    """
        
    # return HTMLResponse(content=index_html, status_code=200)


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests."""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        response_data = chatbot.chat(request.message)
        
        # Post-process the answer to remove unwanted phrases
        answer = response_data.get("answer", "")
        phrases_to_remove = [
            "Based on the provided text, ",
            "Based on the provided texts, ",
            "According to the documents, ",
            "According to the text, ",
            "The context suggests that ",
            "The provided context indicates that ",
            "From the text provided, ",
            "In the provided text, ",
            "Based on the context, ",
            "The text indicates that ",
            "From the documents provided, ",
            "According to the provided information, ",
            "The information provided suggests that ",
            "Based on the information given, ",
            # Add more variations as needed
        ]
        
        # First check for phrases at the beginning of the response
        for phrase in phrases_to_remove:
            if answer.startswith(phrase):
                answer = answer[len(phrase):]
                break
        
        # Then check for these phrases anywhere in the text
        for phrase in phrases_to_remove:
            # Create a regex pattern that's case-insensitive and handles sentence boundaries
            pattern = re.compile(r'\s*' + re.escape(phrase), re.IGNORECASE)
            answer = pattern.sub(' ', answer)
        
        # Update the response data
        response_data["answer"] = answer.strip() # Remove leading/trailing whitespace
        
        return response_data
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent step streaming."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")
            session_id = message_data.get("session_id", "default")
            
            if not message:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Empty message received"
                }))
                continue
            
            # Stream agent steps
            async for step_data in stream_agent_steps(message, session_id):
                await websocket.send_text(json.dumps(step_data))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except:
            pass


async def stream_agent_steps(message: str, session_id: str) -> AsyncGenerator[Dict, None]:
    """Stream agent processing steps in real-time."""
    global chatbot
    
    if chatbot is None:
        yield {
            "type": "error",
            "message": "Chatbot not initialized"
        }
        return
    
    try:
        # Step 1: Thinking
        yield {
            "type": "step",
            "step": "thinking",
            "message": "Analyzing your question...",
            "progress": 10
        }
        await asyncio.sleep(0.1)  # Small delay for UX
        
        # Step 2: Knowledge Graph Search
        yield {
            "type": "step",
            "step": "kg_search",
            "message": "Searching knowledge graph for relevant entities...",
            "progress": 25
        }
        await asyncio.sleep(0.2)
        
        # Step 3: Query Enrichment
        yield {
            "type": "step", 
            "step": "enrichment",
            "message": "Enriching query with domain knowledge...",
            "progress": 40
        }
        await asyncio.sleep(0.2)
        
        # Step 4: Document Retrieval
        yield {
            "type": "step",
            "step": "retrieval",
            "message": "Retrieving relevant research papers...",
            "progress": 65
        }
        await asyncio.sleep(0.3)
        
        # Step 5: Response Generation
        yield {
            "type": "step",
            "step": "generation",
            "message": "Generating comprehensive response...",
            "progress": 85
        }
        await asyncio.sleep(0.2)
        
        # Execute the actual chat (this runs in sync)
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            None, 
            lambda: chatbot.chat(message)
        )
        
        # Step 6: Complete
        yield {
            "type": "step",
            "step": "complete",
            "message": "Response ready!",
            "progress": 100
        }
        await asyncio.sleep(0.1)
        
        # Send final response
        yield {
            "type": "response",
            "answer": response_data.get("answer", ""),
            "sources": response_data.get("sources", [])
        }
        
    except Exception as e:
        logger.error(f"Error in stream_agent_steps: {str(e)}")
        yield {
            "type": "error",
            "message": f"Error processing request: {str(e)}"
        }


def main():
    """Main function to start the server."""
    # Create a .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# LLM Provider Configuration\n")
            f.write("LLM_PROVIDER=google\n\n")
            f.write("# Google API Key - Required if using Google\n")
            f.write("GOOGLE_API_KEY=\n\n")
            f.write("# Azure API Key - Required if using Azure\n")
            f.write("# AZURE_API_KEY=\n\n")
            f.write("# Anthropic API Key - Required if using Claude\n")
            f.write("# ANTHROPIC_API_KEY=\n")
        logger.info("Created .env file. Please add your API keys.")
    
    # Start the server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main() 
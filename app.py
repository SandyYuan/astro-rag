import os
import logging
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request
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


@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup."""
    global chatbot
    
    # Check for API keys based on the selected provider
    provider = os.environ.get("LLM_PROVIDER", LLMProvider.DEFAULT_PROVIDER)
    
    if provider == LLMProvider.PROVIDER_GOOGLE and "GOOGLE_API_KEY" not in os.environ:
        logger.error("GOOGLE_API_KEY environment variable not set. Please set it in .env file.")
    elif provider == LLMProvider.PROVIDER_AZURE and "AZURE_API_KEY" not in os.environ:
        logger.error("AZURE_API_KEY environment variable not set. Please set it in .env file.")
    elif provider == LLMProvider.PROVIDER_CLAUDE and "ANTHROPIC_API_KEY" not in os.environ:
        logger.error("ANTHROPIC_API_KEY environment variable not set. Please set it in .env file.")
    
    # Initialize the chatbot with the configured provider
    try:
        chatbot = AstronomyChatbot(provider=provider)
        logger.info(f"Chatbot initialized successfully with {provider} provider")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page."""
    # Get current provider name for display
    provider = os.environ.get("LLM_PROVIDER", LLMProvider.DEFAULT_PROVIDER).capitalize()
    
    # Create the index.html template content
    index_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professor Wechsler Astronomy Chatbot ({provider})</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .model-info {{
                text-align: center;
                color: #666;
                margin-bottom: 20px;
                font-style: italic;
            }}
            .chat-container {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            #chat-messages {{
                height: 400px;
                overflow-y: auto;
                margin-bottom: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .message {{
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }}
            .user-message {{
                background-color: #e3f2fd;
                text-align: right;
                border-radius: 18px 18px 0 18px;
            }}
            .bot-message {{
                background-color: #f1f1f1;
                border-radius: 18px 18px 18px 0;
            }}
            .sources {{
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
            }}
            .chat-input-container {{
                display: flex;
            }}
            #user-input {{
                flex: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }}
            button {{
                padding: 10px 20px;
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 4px;
                margin-left: 10px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #3367d6;
            }}
            .loading {{
                text-align: center;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>Professor Wechsler Astronomy Chatbot</h1>
        <div class="model-info">Powered by {provider}</div>
        
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
            
            // Function to add a message to the chat
            function addMessage(content, isUser, sources = []) {{
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${{isUser ? 'user-message' : 'bot-message'}}`;
                messageDiv.textContent = content;
                
                // Add sources if available
                if (sources && sources.length > 0) {{
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources';
                    sourcesDiv.textContent = 'Sources: ' + sources.join(', ');
                    messageDiv.appendChild(sourcesDiv);
                }}
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }}
            
            // Function to send message to the server
            async function sendMessage() {{
                const message = userInput.value.trim();
                if (!message) return;
                
                // Add user message to chat
                addMessage(message, true);
                
                // Clear input
                userInput.value = '';
                
                // Add loading indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message bot-message loading';
                loadingDiv.textContent = 'Thinking...';
                messagesContainer.appendChild(loadingDiv);
                
                try {{
                    // Send to server
                    const response = await fetch('/chat', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ message }}),
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Failed to get response');
                    }}
                    
                    const data = await response.json();
                    
                    // Remove loading indicator
                    messagesContainer.removeChild(loadingDiv);
                    
                    // Add bot response
                    addMessage(data.answer, false, data.sources);
                }} catch (error) {{
                    // Remove loading indicator
                    messagesContainer.removeChild(loadingDiv);
                    
                    // Show error message
                    addMessage('Sorry, there was an error processing your request. Please try again.', false);
                    console.error('Error:', error);
                }}
            }}
            
            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            userInput.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter') {{
                    sendMessage();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Write the template to the templates directory
    with open("templates/index.html", "w") as f:
        f.write(index_html)
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests."""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        response = chatbot.chat(request.message)
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
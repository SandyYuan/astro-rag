import os
import logging
import re  # Added for regex pattern matching
import sys  # For sys.executable
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import dotenv
import pkg_resources

# Try importing google.generativeai at startup to verify installation
try:
    import google.generativeai
    google_genai_available = True
    logging.info("Successfully imported google.generativeai at startup")
except ImportError as e:
    google_genai_available = False
    logging.error(f"Failed to import google.generativeai at startup: {e}")

# Check for fallback implementation
try:
    import fallback_genai
    fallback_available = True
    logging.info("Fallback implementation available if needed")
except ImportError:
    fallback_available = False
    logging.warning("Fallback implementation not available")

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

class ApiKeyRequest(BaseModel):
    api_key: str

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup."""
    global chatbot
    
    # Initialize chatbot to None - it will be set when the user provides their API key
    chatbot = None
    logger.info("Chatbot not initialized. Waiting for user to provide API key.")


@app.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    """Set the API key and initialize the chatbot."""
    global chatbot
    
    api_key = request.api_key
    
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "API key is required"}
        )
    
    try:
        # Set the API key in the environment
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize the chatbot with the provided API key
        provider = LLMProvider.PROVIDER_GOOGLE
        chatbot = AstronomyChatbot(provider=provider, api_key=api_key)
        logger.info("Chatbot initialized successfully with user-provided API key")
        
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "API key set successfully"}
        )
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error initializing chatbot: {str(e)}"}
        )


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page."""
    # Get current provider name for display
    provider = "Google"  # Simplified since we only support Google now
    
    # Create the index.html template content
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
            .container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .api-key-container {
                text-align: center;
                margin-bottom: 20px;
            }
            .api-key-container p {
                margin-bottom: 10px;
            }
            #api-key-input {
                width: 70%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            .chat-container {
                display: none; /* Hidden until API key is set */
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
            .alert {
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .alert-success {
                background-color: #d4edda;
                color: #155724;
            }
            .alert-error {
                background-color: #f8d7da;
                color: #721c24;
            }
        </style>
    </head>
    <body>
        <h1>Mini-Risa Chatbot</h1>
        
        <div class="container api-key-container" id="api-key-section">
            <h2>Enter your Google Gemini API Key</h2>
            <p>This chatbot requires a Google Gemini API key to function. Your API key will only be used for this session and is not stored on our servers.</p>
            <p>You can get a Gemini API key from the <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a>.</p>
            <p><strong>Note:</strong> This application includes a lightweight fallback implementation that works even in environments where the official Google Generative AI client can't be installed.</p>
            <div>
                <input 
                    type="password" 
                    id="api-key-input" 
                    placeholder="Enter your Gemini API key here..." 
                    autocomplete="off"
                >
                <button id="set-api-key-button">Set API Key</button>
            </div>
            <div id="api-key-status"></div>
        </div>
        
        <div class="container chat-container" id="chat-section">
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
            const apiKeyInput = document.getElementById('api-key-input');
            const setApiKeyButton = document.getElementById('set-api-key-button');
            const apiKeyStatus = document.getElementById('api-key-status');
            const apiKeySection = document.getElementById('api-key-section');
            const chatSection = document.getElementById('chat-section');
            
            console.log('UI elements initialized');
            
            // Function to handle API key setup
            async function setApiKey() {
                const apiKey = apiKeyInput.value.trim();
                if (!apiKey) {
                    showStatus('Please enter an API key', false);
                    return;
                }
                
                try {
                    // Show a loading message
                    showStatus('Setting up chatbot, please wait...', true);
                    
                    // Send API key to server
                    const response = await fetch('/set-api-key', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ api_key: apiKey }),
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // Show success message
                        showStatus('API key set successfully! Starting chat...', true);
                        
                        // Hide API key section and show chat section after a brief delay
                        setTimeout(() => {
                            apiKeySection.style.display = 'none';
                            chatSection.style.display = 'block';
                        }, 1500);
                    } else {
                        showStatus(`Error: ${data.message}`, false);
                    }
                } catch (error) {
                    showStatus(`Error: ${error.message}`, false);
                }
            }
            
            // Show status message
            function showStatus(message, isSuccess) {
                apiKeyStatus.innerHTML = `<div class="alert ${isSuccess ? 'alert-success' : 'alert-error'}">${message}</div>`;
            }
            
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
            
            // Set API key button event listener
            setApiKeyButton.addEventListener('click', function() {
                console.log('Set API key button clicked');
                setApiKey();
            });
            
            // API key input enter key event listener
            apiKeyInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    console.log('Enter key pressed on API key input');
                    setApiKey();
                }
            });
            
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
    
    # Write the template to the templates directory
    with open("templates/index.html", "w") as f:
        f.write(index_html)
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests."""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized. Please set your API key first.")
    
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
            "Based on the information given, "
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


@app.get("/diagnostics")
async def diagnostics():
    """Diagnostic endpoint to check imports and environment."""
    import sys
    import os
    import pkg_resources
    
    results = {
        "python_version": sys.version,
        "environment_variables": {k: v for k, v in os.environ.items() if not k.startswith('_') and not 'KEY' in k and not 'SECRET' in k},
        "working_directory": os.getcwd(),
        "sys_path": sys.path,
        "installed_packages": [],
        "import_tests": {},
        "fallback_available": False
    }
    
    # Check installed packages
    for pkg in pkg_resources.working_set:
        if any(name in pkg.project_name.lower() for name in ["google", "genai", "langchain", "fastapi", "uvicorn"]):
            results["installed_packages"].append(f"{pkg.project_name}=={pkg.version}")
    
    # Test specific imports
    import_tests = [
        "import google",
        "import google.generativeai",
        "import langchain",
        "import langchain_google_genai",
        "from llm_provider import LLMProvider",
        "from chatbot import AstronomyChatbot"
    ]
    
    for test in import_tests:
        try:
            exec(test)
            results["import_tests"][test] = "Success"
        except Exception as e:
            results["import_tests"][test] = f"Error: {str(e)}"
    
    # Check for fallback_genai
    try:
        import fallback_genai
        results["fallback_available"] = True
        results["import_tests"]["import fallback_genai"] = "Success"
    except ImportError:
        results["import_tests"]["import fallback_genai"] = "Not available"
    
    # Try to create LLMProvider instance with dummy key
    try:
        from llm_provider import LLMProvider
        dummy_provider = LLMProvider(api_key="dummy_key")
        results["llm_provider_init"] = "Success"
    except Exception as e:
        results["llm_provider_init"] = f"Error: {str(e)}"
    
    return results


def main():
    """Main function to start the server."""
    # Create a .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# LLM Provider Configuration\n")
            f.write("LLM_PROVIDER=google\n\n")
            f.write("# Google API Key - Required if using Google\n")
            f.write("GOOGLE_API_KEY=\n\n")
        logger.info("Created .env file. Please add your API keys.")
    
    # Get the port from the environment (for Render) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main() 
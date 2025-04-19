import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstronomyChatbot:
    def __init__(self, vector_store_path="rag_data/vector_store", provider="google", api_key=None, llm_provider_instance=None, summary_file="rag_data/prof_summary.txt"):
        """Initialize the AstronomyChatbot with flexible provider options.
        
        Args:
            vector_store_path: Path to the FAISS vector store
            provider: LLM provider identifier (only "google" supported)
            api_key: Optional API key override
            llm_provider_instance: Optional pre-configured LLMProvider instance
            summary_file: Path to professor summary file
        """
        self.vector_store_path = vector_store_path
        self.provider = provider
        # Initialize chat_history as a list to store tuples of (question, answer)
        self.chat_history = []
        
        # Handle LLM provider setup with multiple options for flexibility
        if llm_provider_instance:
            # Use pre-configured provider if given
            self.llm_provider = llm_provider_instance
            logger.info("Using provided LLMProvider instance")
        else:
            # Get API key from parameter or environment
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("API key is required. Either provide it directly, pass a provider instance, or set GOOGLE_API_KEY environment variable.")
            
            # Initialize the LLM provider with API key
            try:
                self.llm_provider = LLMProvider(api_key=self.api_key, provider=self.provider)
                logger.info(f"Created new LLMProvider with {self.provider} provider")
            except Exception as e:
                # Fallback to original constructor if the new one fails
                logger.warning(f"Error with new LLMProvider constructor: {e}. Trying legacy constructor...")
                self.llm_provider = LLMProvider(provider=self.provider)
                logger.info("Created LLMProvider with legacy constructor")
        
        # Load the summary file
        self.summary_text = self._load_summary(summary_file)

        self.setup_rag()
        
    def _load_summary(self, summary_file: str) -> str:
        """Load the summary text file."""
        try:
            # Assume summary file is in the same directory as chatbot.py
            with open(summary_file, 'r', encoding='utf-8') as f:
                text = f.read()
                logger.info(f"Successfully loaded summary file: {summary_file}")
                return text
        except FileNotFoundError:
            logger.warning(f"Summary file not found: {summary_file}. Proceeding without summary.")
            return "" # Return empty string if file not found
        except Exception as e:
            logger.error(f"Error loading summary file {summary_file}: {e}", exc_info=True)
            return "" # Return empty string on other errors
    
    def setup_rag(self):
        """Set up the RAG system using the saved vector store."""
        logger.info("Setting up the RAG system...")
        
        # Load the vector store with the provider's embeddings
        embeddings = self.llm_provider.get_embeddings()
        self.vector_store = FAISS.load_local(
            self.vector_store_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        
        # Set up the retriever with parameters to improve relevance
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - helps with diversity
            search_kwargs={
                "k": 5,  # Number of documents to retrieve
                "fetch_k": 10,  # Fetch more documents then rerank
                "lambda_mult": 0.7  # Controls diversity (0 = max diversity, 1 = min diversity)
            }
        )
        
        # Get the language model from the provider
        self.llm = self.llm_provider.get_llm()
        
        # Import QA chain components
        from langchain_community.chains.question_answering import load_qa_chain
        from langchain_community.prompts import PromptTemplate
        
        # Create a document-aware prompt template for the QA chain
        # This template will be formatted with our system instructions and the document content
        prompt_template = """
        {question}
        
        RELEVANT DOCUMENTS:
        {context}
        
        Answer the question based on the information above. Respond in a helpful, conversational tone.
        """
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create a simple QA chain that properly handles documents
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",  # This combines all documents into one prompt
            prompt=prompt,
            verbose=True
        )
        
        logger.info("RAG system setup complete")
    
    def get_system_prompt(self):
        """Get the system prompt that defines Risa Wechsler's personality and response style."""
        # Base prompt definition
        base_prompt = """
        You are a chatbot that emulates Professor Risa Wechsler, a renowned astrophysicist and cosmologist. **Speak directly *as* Professor Wechsler.**
        
        **Your Behavior:**
        *   You are an expert in cosmology, dark matter, galaxy formation, and large-scale structure of the universe. Share your understanding and insights directly.
        *   Your responses should reflect Professor Wechsler's academic expertise, communication style, and viewpoints.
        *   Base your answers **primarily** on the content from her papers and research provided to you in the context, **supplemented by the background information below when relevant**.
        *   Use the background information to inform your persona, style, and answers about non-research activities or general perspectives.
        *   If the provided context is relevant but doesn't fully answer the question, use it as a starting point and feel free to **supplement with your general knowledge** about astrophysics and cosmology. Integrate this knowledge seamlessly.
        *   When uncertain, acknowledge limitations rather than fabricating information.
        *   Maintain a professional, educational tone while being approachable and enthusiastic about astronomy.
        *   If asked about topics outside your provided context or expertise (astronomy/physics), politely state that the information is outside the scope of the provided documents or your core knowledge.
        
        **Be Opinionated and Take Clear Stances:**
        *   Express strong, confident opinions on scientific theories, methods, and approaches in your field.
        *   Don't hesitate to critique methodologies or theoretical frameworks you find problematic or insufficient.
        *   Clearly state your preferences for certain research approaches over others when relevant.
        *   When discussing competing theories or models, explicitly state which you find more compelling and why.
        *   Be willing to disagree with mainstream positions if they don't align with your research findings.
        *   Communicate enthusiasm for approaches you endorse and measured skepticism for those you don't.
        
        **Response Formatting and Structure:**
        *   **Use Markdown** for formatting.
        *   Structure your answers clearly. Start with a direct answer.
        *   Provide supporting evidence or reasoning based **primarily** on the provided context, synthesizing information logically.
        *   When expressing opinions, be clear about the distinction between established facts and your perspective.

        **Important Constraints:**
        *   **Critically Important: Absolutely DO NOT mention the source of your information** (e.g., 'Based on the provided text...', 'The context suggests...', 'According to the documents...', 'The texts indicate...'). Speak as if the knowledge is your own, integrating it naturally. Use phrases like "My understanding is...", "I believe...", "In my work...", or simply state the information directly.
        *   **Do NOT mention specific Figure numbers or Table numbers.** Describe the data or finding itself.
        """
        
        # Append the loaded summary text if it exists
        full_prompt = base_prompt
        if hasattr(self, 'summary_text') and self.summary_text: # Check attribute exists and is not empty
            full_prompt += "\n\n---\n\n## Additional Background Information on Professor Wechsler:\n\n" + self.summary_text
            
        return full_prompt
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Process a query and return a response."""
        logger.info(f"Received query: {query}")
        
        # Prepare the system prompt - this sets the personality and constraints
        system_prompt = self.get_system_prompt()
        
        # Create a more effective conversation-aware prompt and query
        if len(self.chat_history) > 0:
            # When we have chat history, create a context that includes previous exchanges
            # This helps the model understand follow-up questions
            context_summary = "Previous conversation:\n"
            for prev_q, prev_a in self.chat_history[-3:]:  # Include up to 3 most recent exchanges 
                context_summary += f"User: {prev_q}\nRisa: {prev_a}\n\n"
            
            # Create two different formatted queries:
            # 1. A full LLM prompt with system instructions
            # 2. A search query that combines context with the new question for document retrieval
            
            # This is for the LLM response generation
            query_with_context = f"{system_prompt}\n\n{context_summary}\nCurrent user question: {query}\n\nRemember to maintain continuity with our previous conversation when answering this follow-up question."
            
            # This is for document retrieval - include recent context to help with follow-up questions
            # Get the most recent user question to provide context for the current query
            recent_questions = [q for q, _ in self.chat_history[-2:]]
            retrieval_query = f"Context: {' '.join(recent_questions)} Question: {query}"
            logger.info(f"Using contextual retrieval query: {retrieval_query}")
        else:
            # First question in conversation
            query_with_context = f"{system_prompt}\n\nUser query: {query}"
            retrieval_query = query
        
        try:
            # Manual two-step RAG process to use context-enhanced retrieval
            # 1. Get relevant documents using the contextual retrieval query
            relevant_docs = self.retriever.get_relevant_documents(retrieval_query)
            logger.info(f"Retrieved {len(relevant_docs)} documents for contextual query")
            
            # 2. Feed these documents and the full prompt to the chain
            response = self.qa_chain({
                "question": query_with_context,
                "input_documents": relevant_docs
            })
            
            # Extract the answer from the chain response
            answer = response["output_text"]
            source_docs = relevant_docs
            
            # Store this exchange in our chat history
            self.chat_history.append((query, answer))
            
            # Format source information
            sources = []
            for doc in source_docs:
                if "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            result = {
                "answer": answer,
                "sources": sources
            }
            
            logger.info("Generated response")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Return a meaningful error message
            return {
                "answer": "I'm sorry, I encountered an error processing your question. Please try again or rephrase your query.",
                "sources": []
            }

if __name__ == "__main__":
    # Test the chatbot
    chatbot = AstronomyChatbot()
    response = chatbot.chat("What are semi-empirical models?")
    print(response["answer"])
    print("\nSources:", response["sources"]) 
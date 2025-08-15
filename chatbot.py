import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AstronomyChatbot:
    def __init__(
        self,
        vector_store_path: str = "rag_data/vector_store",
        api_key: Optional[str] = None,
        llm_provider_instance: Optional[LLMProvider] = None,
        summary_file: str = "rag_data/prof_summary.txt",
        retrieval_mode: Optional[str] = None,  # Legacy parameter, ignored
    ) -> None:
        """Initialize the AstronomyChatbot with KG-enriched sequential retrieval.

        Args:
            vector_store_path: Path to the FAISS vector store
            api_key: Optional API key override
            llm_provider_instance: Optional pre-configured LLMProvider instance
            summary_file: Path to professor summary file
            retrieval_mode: Legacy parameter, ignored (system always uses KG-enriched sequential)
        """
        self.vector_store_path = vector_store_path

        # Always use sequential KG-enriched retrieval (only mode available)
        self.retrieval_mode = "kg_enriched_sequential"
        logger.info(f"Using sequential KG-enriched retrieval (only available mode)")
        
        # Always use agent mode for conversation capabilities
        self.chat_mode = "agent"
        logger.info(f"Chat mode: {self.chat_mode} (conversation memory enabled)")
        
        # Handle LLM provider setup
        if llm_provider_instance:
            # Use pre-configured provider if given
            self.llm_provider = llm_provider_instance
            logger.info("Using provided LLMProvider instance")
        else:
            # Get API key from parameter or environment
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("API key is required. Either provide it directly, pass a provider instance, or set GOOGLE_API_KEY environment variable.")
            # Initialize the LLM provider with API key (no fallback)
            self.llm_provider = LLMProvider(api_key=self.api_key)
            logger.info("Created LLMProvider")
        
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
    
    def setup_rag(self) -> None:
        """Set up the KG-enriched sequential retrieval pipeline."""
        logger.info("Setting up the KG-enriched sequential retrieval system...")

        # Initialize FAISS vector store
        embeddings = self.llm_provider.get_embeddings()
        self.vector_store = FAISS.load_local(
            self.vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")

        # Set up FAISS retriever
        self.faiss_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,  # Get more candidates for better quality
                "lambda_mult": 0.7,
            },
        )
        logger.info("FAISS retriever configured (MMR)")

        # Initialize Neo4j GraphRetriever
        from graph_rag.neo4j_client import GraphRetriever
        self.graph_retriever = GraphRetriever(k=5)
        logger.info("Neo4j GraphRetriever initialized")
        
        # Initialize KG-enriched sequential pipeline
        from retrieval.kg_filter import KGQueryFilter
        from retrieval.kg_enriched_retrieval import KGEnrichedRetriever
        
        self.kg_filter = KGQueryFilter(self.llm_provider)
        self.kg_enriched_retriever = KGEnrichedRetriever(
            graph_retriever=self.graph_retriever,
            vector_retriever=self.faiss_retriever,
            kg_filter=self.kg_filter
        )
        logger.info("KG-enriched sequential retrieval pipeline initialized")

        # Configure the language model (generation remains the same across modes)
        self.llm = self.llm_provider.get_llm(model_name="gemini-2.5-flash")

        # Import QA chain components
        from langchain.chains.question_answering import load_qa_chain
        from langchain.prompts import PromptTemplate

        # Create a document-aware prompt template for the QA chain
        prompt_template = """
        {question}

        RELEVANT DOCUMENTS:
        {context}

        Answer the question based on the information above. Respond in a helpful, conversational tone.
        """

        # Create the prompt template
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create a simple QA chain that properly handles documents
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt,
            verbose=True,
        )

        logger.info("RAG system setup complete")
    
    def _create_standalone_question(self, query: str) -> str:
        """
        Convert follow-up question to standalone question using LLM.
        Uses industry-standard query condensation pattern.
        
        Args:
            query: Current user question (may contain pronouns/references)
        
        Returns:
            Standalone question with necessary context included
        """
        if not hasattr(self, 'chat_history') or not self.chat_history:
            return query  # First question is already standalone
        
        # Get recent conversation context (last 2 exchanges max)
        recent_history = []
        for q, a in self.chat_history[-2:]:  # Last 2 exchanges to avoid token bloat
            recent_history.append(f"Human: {q}")
            # Truncate long answers to keep context focused
            answer_snippet = a[:300] + "..." if len(a) > 300 else a
            recent_history.append(f"Assistant: {answer_snippet}")
        
        history_text = "\n".join(recent_history)
        
        condense_prompt = f"""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context.

Make the standalone question clear and specific, resolving any pronouns or references to previous topics.

Conversation History:
{history_text}

Follow-up Question: {query}

Standalone Question:"""

        # Use fast, cheap model for query condensation
        condenser_llm = self.llm_provider.get_llm(
            model_name="gemini-2.5-flash",
            temperature=0.1  # Low temperature for consistent rewriting
        )
        
        standalone_question = condenser_llm.invoke(condense_prompt).strip()
        
        # Validate that LLM produced a meaningful result
        if not standalone_question or len(standalone_question.strip()) == 0:
            logger.error("Query condensation produced empty result - this indicates an LLM issue")
            raise ValueError("Query condensation failed: empty response from LLM")
        
        logger.info(f"Query condensation: '{query}' → '{standalone_question}'")
        return standalone_question

    
    def _intelligent_retrieval(self, query: str, retrieval_query: str) -> List[Any]:
        """
        KG-enriched sequential retrieval pipeline.
        
        Pipeline: KG → LLM filter → query enrichment → vector search
        
        Args:
            query: Original query 
            retrieval_query: Legacy parameter (ignored)
        
        Returns:
            List of relevant documents from KG-enriched vector search
        """
        logger.info("Using KG-enriched sequential retrieval pipeline")
        standalone_question = self._create_standalone_question(query)
        return self.kg_enriched_retriever.get_relevant_documents(standalone_question)
    def _setup_react_agent(self):
        """Set up the LangGraph ReAct agent with proper checkpointer."""
        if not hasattr(self, '_agent_executor'):
            # Create document retrieval tool using KG-enriched sequential pipeline
            def retrieval_func(query: str) -> str:
                """Search research papers and documents using KG-enriched retrieval."""
                try:
                    docs = self._intelligent_retrieval(query, query)
                    
                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs[:3]])
                        sources = [doc.metadata.get("source", "Unknown") for doc in docs[:3]]
                        return f"Context: {context}\nSources: {', '.join(sources)}"
                    else:
                        return "No relevant documents found."
                        
                except Exception as e:
                    logger.error(f"Retrieval tool error: {e}")
                    return f"Error retrieving documents: {e}"
            
            tools = [Tool(
                name="document_search",
                func=retrieval_func,
                description="Search through research papers and academic documents for scientific concepts, research findings, measurements, and academic topics."
            )]
            
            # Create checkpointer for conversation memory
            checkpointer = MemorySaver()
            
            # Create a proper LangChain model for LangGraph agent (needs bind_tools support)
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            agent_model = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=self.llm_provider.api_key
            )
            
            # Create the ReAct agent with proper conversation memory management
            # Using the built-in create_react_agent with checkpointer
            self._agent_executor = create_react_agent(
                model=agent_model,
                tools=tools,
                checkpointer=checkpointer,
                # Optional: Add message trimming to prevent context overflow
                # pre_model_hook=self._trim_messages_hook
            )
            
            logger.info("ReAct agent initialized with LangGraph MemorySaver checkpointer")
    
    def _chat_agent_mode(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced chat with LangGraph ReAct agent and proper conversation memory.
        
        Args:
            query: User query
            session_id: Optional session ID for conversation continuity
        
        Returns:
            Dict with 'answer', 'sources', and optional 'reasoning_trace'
        """
        self._setup_react_agent()
        
        try:
            # Use session_id as thread_id for LangGraph conversation memory
            thread_id = session_id or "default"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute the ReAct agent - conversation memory is handled automatically
            # by the built-in create_react_agent with checkpointer
            response = self._agent_executor.invoke(
                {"messages": [("user", query)]}, 
                config=config
            )
            
            # Extract the final answer from the response
            messages = response.get("messages", [])
            if messages:
                # Get the last AI message as the answer
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        answer = msg.content
                        break
                else:
                    answer = "No response generated."
            else:
                answer = "No response generated."
            
            # For now, we'll extract sources and reasoning from the messages
            # This is a simplified approach since the built-in agent structure is different
            sources = []
            reasoning_trace = []
            
            # Look through messages for tool calls and responses
            for msg in messages:
                if hasattr(msg, 'type'):
                    if msg.type == 'tool':
                        reasoning_trace.append(f"Tool Call: {getattr(msg, 'name', 'unknown')}")
                        reasoning_trace.append(f"Result: {msg.content[:200]}...")
                        
                        # Extract sources from tool results
                        if "Sources:" in msg.content:
                            obs_sources = msg.content.split("Sources:")[-1].strip()
                            for source in obs_sources.split(", "):
                                if source and source not in sources:
                                    sources.append(source)
            
            result = {
                "answer": answer,
                "sources": sources,
                "reasoning_trace": reasoning_trace
            }
            
            logger.info(f"ReAct agent completed with {len(messages)} messages")
            return result
            
        except Exception as e:
            logger.error(f"Error in ReAct agent mode: {e}", exc_info=True)
            return {
                "answer": "I encountered an error processing your question. Please try again or rephrase your query.",
                "sources": [],
                "reasoning_trace": [f"Error: {e}"]
            }
    
    def chat(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query and return a response using ReAct agent with conversation memory."""
        logger.info(f"Received query: {query}")
        
        # Always use agent mode for enhanced conversation capabilities
        return self._chat_agent_mode(query, session_id)

if __name__ == "__main__":
    # Test the chatbot
    chatbot = AstronomyChatbot()
    response = chatbot.chat("What are semi-empirical models?")
    print(response["answer"])
    print("\nSources:", response["sources"]) 
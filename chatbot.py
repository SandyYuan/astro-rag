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
        retrieval_mode: Optional[str] = None,
    ) -> None:
        """Initialize the AstronomyChatbot.

        Args:
            vector_store_path: Path to the FAISS vector store (used when RAG_MODE=faiss)
            api_key: Optional API key override
            llm_provider_instance: Optional pre-configured LLMProvider instance
            summary_file: Path to professor summary file
            retrieval_mode: Retrieval backend selection; one of {"faiss", "neo4j"}. Defaults to env RAG_MODE or "faiss".
        """
        self.vector_store_path = vector_store_path

        # Determine retrieval mode (no fallbacks). Only accept explicit values.
        env_mode = os.environ.get("RAG_MODE", "faiss").strip().lower()
        selected_mode = (retrieval_mode or env_mode).strip().lower()
        if selected_mode not in {"faiss", "neo4j", "dual"}:
            raise ValueError("Invalid RAG_MODE. Expected 'faiss', 'neo4j', or 'dual'.")
        self.retrieval_mode = selected_mode
        logger.info(f"Retrieval mode set to: {self.retrieval_mode}")
        
        # Determine chat mode for enhanced capabilities
        self.chat_mode = os.environ.get("CHAT_MODE", "legacy").strip().lower()
        if self.chat_mode not in {"legacy", "agent"}:
            raise ValueError("Invalid CHAT_MODE. Expected 'legacy' or 'agent'.")
        logger.info(f"Chat mode set to: {self.chat_mode}")
        
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
        """Set up the retrieval backend and QA chain.

        When in FAISS mode, loads the local vector store and configures an MMR retriever.
        When in Neo4j mode, initializes the graph-backed GraphRetriever.
        When in dual mode, initializes both retrievers for fusion.
        """
        logger.info("Setting up the RAG system...")

        if self.retrieval_mode == "faiss":
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
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 20,  # Increased to get more candidates for filtering
                    "lambda_mult": 0.7,
                },
            )
            logger.info("FAISS retriever configured (MMR)")
        elif self.retrieval_mode == "neo4j":
            # Neo4j GraphRAG retriever
            from graph_rag.neo4j_client import GraphRetriever

            self.retriever = GraphRetriever(k=5)
            logger.info("Neo4j GraphRetriever initialized")
        else:  # dual mode
            # Initialize both retrievers for fusion
            embeddings = self.llm_provider.get_embeddings()
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")

            self.faiss_retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 20,  # Increased to get more candidates for filtering
                    "lambda_mult": 0.7,
                },
            )
            logger.info("FAISS retriever configured (MMR)")

            from graph_rag.neo4j_client import GraphRetriever
            self.graph_retriever = GraphRetriever(k=5)
            logger.info("Neo4j GraphRetriever initialized")
            
            # No single retriever in dual mode - we'll handle retrieval differently
            self.retriever = None
            logger.info("Dual retrieval mode configured")

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
    

    
    def _dual_retrieval_with_fusion(self, query: str, retrieval_query: str) -> List[Any]:
        """
        Perform dual retrieval from both FAISS and Neo4j, then fuse results.
        Uses industry-standard query condensation for consistent retrieval.
        
        Args:
            query: Original query 
            retrieval_query: Legacy parameter (ignored, we use standalone question)
        
        Returns:
            List of fused and budget-enforced documents
        """
        from retrieval.fusion import (
            reciprocal_rank_fusion, 
            normalize_scores, 
            enforce_token_budget
        )
        
        # Step 1: Create standalone question for consistent retrieval
        standalone_question = self._create_standalone_question(query)
        
        # Step 2: Retrieve from both sources using the same standalone question
        faiss_docs = []
        neo4j_docs = []
        
        try:
            # FAISS retrieval with standalone question
            raw_faiss_docs = self.faiss_retriever.get_relevant_documents(standalone_question)
            # Apply content quality filtering to FAISS results
            from retrieval.content_filter import filter_quality_documents
            faiss_docs = filter_quality_documents(raw_faiss_docs, min_tokens=30)
            logger.info(f"FAISS retrieved {len(raw_faiss_docs)} documents, {len(faiss_docs)} after quality filtering")
        except Exception as e:
            logger.warning(f"FAISS retrieval failed: {e}")
        
        try:
            # Neo4j retrieval with standalone question  
            neo4j_docs = self.graph_retriever.get_relevant_documents(standalone_question)
            logger.info(f"Neo4j retrieved {len(neo4j_docs)} documents")
        except Exception as e:
            logger.warning(f"Neo4j retrieval failed: {e}")
        
        # If both failed, return empty list
        if not faiss_docs and not neo4j_docs:
            logger.warning("Both retrievers failed - returning empty results")
            return []
        
        # Prepare scored document lists for fusion
        faiss_scored = []
        neo4j_scored = []
        
        # FAISS docs: extract scores from metadata or use rank-based scoring
        for i, doc in enumerate(faiss_docs):
            score = doc.metadata.get("score")
            if score is None:
                # Use similarity score if available in different metadata key
                score = doc.metadata.get("similarity", 0.9 - i * 0.1)  # Fallback scoring
            faiss_scored.append((doc, score))
        
        # Neo4j docs: use rank-based scoring (no similarity scores available)
        for i, doc in enumerate(neo4j_docs):
            neo4j_scored.append((doc, None))  # Will be normalized by rank
        
        # Normalize scores for each retriever type
        if faiss_scored:
            faiss_normalized = normalize_scores(faiss_scored, method="minmax")
        else:
            faiss_normalized = []
            
        if neo4j_scored:
            neo4j_normalized = normalize_scores(neo4j_scored, method="rank")
        else:
            neo4j_normalized = []
        
        # Prepare ranked lists for RRF (convert to rank-based)
        faiss_ranked = [(doc, i) for i, (doc, _) in enumerate(faiss_normalized)]
        neo4j_ranked = [(doc, i) for i, (doc, _) in enumerate(neo4j_normalized)]
        
        # Apply Reciprocal Rank Fusion
        ranked_lists = []
        if faiss_ranked:
            ranked_lists.append(faiss_ranked)
        if neo4j_ranked:
            ranked_lists.append(neo4j_ranked)
        
        fused_results = reciprocal_rank_fusion(ranked_lists, k=60)
        
        # Extract documents from fused results
        fused_docs = [doc for doc, _ in fused_results]
        
        # Apply token budget enforcement (default 3000 tokens)
        token_budget = int(os.environ.get("FUSION_TOKEN_BUDGET", "3000"))
        diversity_factor = float(os.environ.get("FUSION_DIVERSITY_FACTOR", "0.5"))
        
        final_docs = enforce_token_budget(
            fused_docs, 
            budget=token_budget,
            min_docs=2,  # Ensure at least 2 docs even if over budget
            diversity_factor=diversity_factor
        )
        
        logger.info(f"Fusion pipeline: {len(faiss_docs)} FAISS + {len(neo4j_docs)} Neo4j → "
                   f"{len(fused_docs)} fused → {len(final_docs)} final (budget: {token_budget} tokens)")
        
        return final_docs
    

    

    

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
    
    def _setup_react_agent(self):
        """Set up the LangGraph ReAct agent with proper checkpointer."""
        if not hasattr(self, '_agent_executor'):
            # Create document retrieval tool
            def retrieval_func(query: str) -> str:
                """Search research papers and documents."""
                try:
                    if self.retrieval_mode == "dual":
                        docs = self._dual_retrieval_with_fusion(query, query)
                    else:
                        raw_docs = self.retriever.get_relevant_documents(query)
                        
                        if self.retrieval_mode == "faiss":
                            from retrieval.content_filter import filter_quality_documents
                            docs = filter_quality_documents(raw_docs, min_tokens=30)
                        else:
                            docs = raw_docs
                    
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
        """Process a query and return a response."""
        logger.info(f"Received query: {query}")
        
        # Route to appropriate chat mode
        if self.chat_mode == "agent":
            return self._chat_agent_mode(query, session_id)
        
        # Legacy mode - simple retrieval without conversation memory
        system_prompt = self.get_system_prompt()
        query_with_context = f"{system_prompt}\n\nUser query: {query}"
        
        try:
            # Simple retrieval based on mode
            if self.retrieval_mode == "dual":
                relevant_docs = self._dual_retrieval_with_fusion(query, query)
            else:
                raw_docs = self.retriever.get_relevant_documents(query)
                
                if self.retrieval_mode == "faiss":
                    from retrieval.content_filter import filter_quality_documents
                    relevant_docs = filter_quality_documents(raw_docs, min_tokens=30)
                else:
                    relevant_docs = raw_docs
            
            # Generate response using QA chain
            response = self.qa_chain.invoke({
                "question": query_with_context,
                "input_documents": relevant_docs,
            })
            
            answer = response["output_text"]
            
            # Extract sources
            sources = []
            for doc in relevant_docs:
                if "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in legacy mode: {e}", exc_info=True)
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
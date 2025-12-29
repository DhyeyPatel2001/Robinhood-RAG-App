"""
Robinhood RAG - Streamlit Chat Interface

A conversational interface for querying Robinhood support documentation
using hybrid search and agentic RAG.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import settings

# Page configuration
st.set_page_config(
    page_title="Robinhood Support Assistant",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Source card styling */
    .source-card {
        background: rgba(0, 200, 83, 0.1);
        border-left: 3px solid #00c853;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .main-header h1 {
        color: #00c853;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .status-ready {
        background: rgba(0, 200, 83, 0.2);
        color: #00c853;
    }
    
    .status-error {
        background: rgba(255, 82, 82, 0.2);
        color: #ff5252;
    }
    
    /* Thoughts container */
    .thoughts-container {
        background: rgba(100, 100, 255, 0.1);
        border-left: 3px solid #6b7fff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        font-style: italic;
    }
    
    .thoughts-header {
        color: #6b7fff;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-style: normal;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
        st.session_state.agent_error = None
    
    if "phoenix_enabled" not in st.session_state:
        st.session_state.phoenix_enabled = True  # Enabled by default
    
    if "show_thoughts" not in st.session_state:
        st.session_state.show_thoughts = True
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    if "phoenix_initialized" not in st.session_state:
        st.session_state.phoenix_initialized = False


def initialize_agent():
    """Initialize the RAG agent."""
    if st.session_state.agent is not None:
        return True
    
    try:
        # Check for API key
        if not settings.LLM_API_KEY:
            st.session_state.agent_error = "LLM API key not set. Please configure in sidebar."
            return False
        
        # IMPORTANT: Initialize Phoenix BEFORE creating LLM client
        # This is required for OpenAI instrumentation to work
        if st.session_state.phoenix_enabled:
            initialize_phoenix()
        
        # Import and initialize components
        from src.llm.agent import RAGAgent
        from src.indexing.hybrid_search import HybridSearch
        from src.llm.client import LLMClient
        
        # Initialize with progress
        with st.spinner("Initializing RAG agent..."):
            client = LLMClient()
            search = HybridSearch()
            agent = RAGAgent(
                llm_client=client,
                hybrid_search=search,
            )
            
            st.session_state.agent = agent
            st.session_state.agent_error = None
        
        return True
        
    except Exception as e:
        st.session_state.agent_error = str(e)
        return False


@st.cache_resource
def _init_phoenix_once():
    """Initialize Phoenix telemetry (cached - runs only once)."""
    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.openai import OpenAIInstrumentor
        
        # Launch Phoenix app
        px.launch_app()
        print("Phoenix UI launched at http://localhost:6006")
        
        # Register OTEL tracer
        tracer_provider = register(project_name="robinhood-rag")
        
        # Instrument OpenAI client BEFORE any OpenAI client is created
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        print("Phoenix telemetry initialized")
        return True
        
    except ImportError as e:
        print(f"Phoenix dependencies not installed: {e}")
        return False
    except Exception as e:
        print(f"Failed to initialize Phoenix: {e}")
        return False


def initialize_phoenix():
    """Initialize Phoenix telemetry if enabled."""
    if not st.session_state.phoenix_enabled:
        return False
    return _init_phoenix_once()


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Configuration
        st.markdown("### 🔑 API Settings")
        
        api_key = st.text_input(
            "LLM API Key",
            value=settings.LLM_API_KEY or "",
            type="password",
            help="Your API key for the LLM service",
        )
        
        if api_key and api_key != settings.LLM_API_KEY:
            os.environ["LLM_API_KEY"] = api_key
            settings.LLM_API_KEY = api_key
            st.session_state.agent = None  # Force re-initialization
        
        base_url = st.text_input(
            "Base URL",
            value=settings.LLM_BASE_URL,
            help="API base URL (e.g., https://api.deepseek.com)",
        )
        
        if base_url != settings.LLM_BASE_URL:
            os.environ["LLM_BASE_URL"] = base_url
            settings.LLM_BASE_URL = base_url
            st.session_state.agent = None
        
        model = st.text_input(
            "Model",
            value=settings.LLM_MODEL,
            help="Model name to use",
        )
        
        if model != settings.LLM_MODEL:
            os.environ["LLM_MODEL"] = model
            settings.LLM_MODEL = model
            st.session_state.agent = None
        
        st.markdown("---")
        
        # Display Settings
        st.markdown("### 📊 Display Options")
        
        st.session_state.show_sources = st.checkbox(
            "Show Sources",
            value=st.session_state.show_sources,
            help="Display source documents used for answers",
        )
        
        st.session_state.show_thoughts = st.checkbox(
            "Show Reasoning",
            value=st.session_state.show_thoughts,
            help="Display the agent's thinking process",
        )
        
        st.markdown("---")
        
        # Telemetry
        st.markdown("### 📈 Observability")
        
        phoenix_enabled = st.checkbox(
            "Enable Phoenix Telemetry",
            value=st.session_state.phoenix_enabled,
            help="Enable Arize Phoenix for LLM tracing",
        )
        
        if phoenix_enabled != st.session_state.phoenix_enabled:
            st.session_state.phoenix_enabled = phoenix_enabled
            if phoenix_enabled:
                initialize_phoenix()
        
        if st.session_state.phoenix_enabled:
            st.markdown(
                "🔗 [Open Phoenix Dashboard](http://localhost:6006)",
                unsafe_allow_html=True,
            )
        
        st.markdown("---")
        
        # Actions
        st.markdown("### 🔧 Actions")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        if st.button("🔄 Reinitialize Agent", use_container_width=True):
            st.session_state.agent = None
            st.session_state.agent_error = None
            st.rerun()
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📡 Status")
        
        if st.session_state.agent is not None:
            st.success("✅ Agent Ready")
        elif st.session_state.agent_error:
            st.error(f"❌ {st.session_state.agent_error}")
        else:
            st.info("⏳ Agent not initialized")


def render_sources(sources: list):
    """Render source documents."""
    if not sources or not st.session_state.show_sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(
                f"""<div class="source-card">
                    <strong>{i}. {source.get('title', 'Unknown')}</strong><br>
                    <a href="{source.get('url', '#')}" target="_blank">{source.get('url', '')}</a>
                </div>""",
                unsafe_allow_html=True,
            )


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏹 Robinhood Support Assistant</h1>
        <p>Ask questions about Robinhood's products, services, and features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize agent if needed
    if st.session_state.agent is None and settings.LLM_API_KEY:
        initialize_agent()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Robinhood..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if agent is ready
        if st.session_state.agent is None:
            if not initialize_agent():
                with st.chat_message("assistant"):
                    st.error(
                        "Agent not initialized. "
                        "Please configure your API key in the sidebar."
                    )
                return
        
        # Generate response with streaming
        with st.chat_message("assistant"):
            try:
                # Containers for streaming
                thoughts_placeholder = st.empty()
                answer_placeholder = st.empty()
                sources_placeholder = st.empty()
                
                thoughts_text = ""
                answer_text = ""
                sources = []
                in_thinking = False
                in_answer = False
                
                # Stream the response
                for event in st.session_state.agent.chat_stream(prompt):
                    event_type = event.get("type", "")
                    content = event.get("content", "")
                    
                    if event_type == "thinking_start":
                        in_thinking = True
                        if st.session_state.show_thoughts:
                            thoughts_placeholder.markdown(
                                '<div class="thoughts-container"><div class="thoughts-header">🤔 Thinking...</div></div>',
                                unsafe_allow_html=True
                            )
                    
                    elif event_type == "thinking" and in_thinking:
                        thoughts_text += content
                        if st.session_state.show_thoughts:
                            thoughts_placeholder.markdown(
                                f'<div class="thoughts-container"><div class="thoughts-header">🤔 Thinking...</div>{thoughts_text}</div>',
                                unsafe_allow_html=True
                            )
                    
                    elif event_type == "thinking_end":
                        in_thinking = False
                    
                    elif event_type == "search":
                        answer_placeholder.markdown(f"🔍 Searching: *{content}*")
                    
                    elif event_type == "results":
                        answer_placeholder.markdown(f"📚 {content}")
                    
                    elif event_type == "answer_start":
                        in_answer = True
                        answer_text = ""
                    
                    elif event_type == "answer" and in_answer:
                        answer_text += content
                        answer_placeholder.markdown(answer_text)
                    
                    elif event_type == "answer_end":
                        in_answer = False
                    
                    elif event_type == "complete":
                        answer_text = event.get("content", answer_text)
                        sources = event.get("sources", [])
                        all_thoughts = event.get("thoughts", [])
                        
                        # Clear placeholders and render final content
                        thoughts_placeholder.empty()
                        
                        # Show thoughts in expander
                        if all_thoughts and st.session_state.show_thoughts:
                            with thoughts_placeholder.expander("🤔 View Reasoning", expanded=False):
                                for i, thought in enumerate(all_thoughts, 1):
                                    st.markdown(f"**Step {i}:**\n{thought}")
                        
                        # Show final answer
                        answer_placeholder.markdown(answer_text)
                        
                        # Show sources
                        with sources_placeholder:
                            render_sources(sources)
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


if __name__ == "__main__":
    main()

"""
Arize Phoenix telemetry integration.

Provides observability for LLM calls, tool invocations, and the full RAG flow.
Uses OpenTelemetry for instrumentation.
"""

import os
from typing import Optional
from contextlib import contextmanager

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Global state for Phoenix
_phoenix_initialized = False
_tracer = None


def initialize_phoenix(
    project_name: str = "robinhood-rag",
    launch_app: bool = True,
) -> bool:
    """Initialize Arize Phoenix for telemetry.
    
    Args:
        project_name: Name for the Phoenix project
        launch_app: Whether to launch the Phoenix web UI
        
    Returns:
        True if initialization successful
    """
    global _phoenix_initialized, _tracer
    
    if _phoenix_initialized:
        return True
    
    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.openai import OpenAIInstrumentor
        
        # Launch Phoenix app if requested
        if launch_app:
            px.launch_app()
            print("Phoenix UI launched at http://localhost:6006")
        
        # Register OTEL tracer
        tracer_provider = register(project_name=project_name)
        _tracer = tracer_provider.get_tracer(__name__)
        
        # Instrument OpenAI client
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        _phoenix_initialized = True
        print(f"Phoenix telemetry initialized for project: {project_name}")
        
        return True
        
    except ImportError as e:
        print(f"Phoenix dependencies not installed: {e}")
        print("Install with: pip install arize-phoenix openinference-instrumentation-openai")
        return False
    except Exception as e:
        print(f"Error initializing Phoenix: {e}")
        return False


def get_tracer():
    """Get the Phoenix tracer for custom spans.
    
    Returns:
        OpenTelemetry tracer or None if not initialized
    """
    global _tracer
    return _tracer


@contextmanager
def trace_span(name: str, attributes: Optional[dict] = None):
    """Create a custom trace span.
    
    Args:
        name: Span name
        attributes: Optional span attributes
        
    Yields:
        The span object
    """
    global _tracer
    
    if _tracer is None:
        # No-op if not initialized
        yield None
        return
    
    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def trace_search(query: str, results: list, source: str = "hybrid"):
    """Log a search operation to Phoenix.
    
    Args:
        query: Search query
        results: Search results
        source: Search source (vector, keyword, hybrid)
    """
    global _tracer
    
    if _tracer is None:
        return
    
    with _tracer.start_as_current_span("search") as span:
        span.set_attribute("search.query", query)
        span.set_attribute("search.source", source)
        span.set_attribute("search.num_results", len(results))
        
        if results:
            span.set_attribute(
                "search.top_score",
                results[0].get("rrf_score", results[0].get("score", 0))
            )


def trace_agent_iteration(
    iteration: int,
    action: str,
    details: Optional[str] = None,
):
    """Log an agent iteration to Phoenix.
    
    Args:
        iteration: Iteration number
        action: Action taken (tool_call, final_answer, etc.)
        details: Additional details
    """
    global _tracer
    
    if _tracer is None:
        return
    
    with _tracer.start_as_current_span(f"agent_iteration_{iteration}") as span:
        span.set_attribute("agent.iteration", iteration)
        span.set_attribute("agent.action", action)
        if details:
            span.set_attribute("agent.details", details)


class PhoenixTracedAgent:
    """Wrapper to add Phoenix tracing to RAGAgent."""
    
    def __init__(self, agent):
        """Initialize with a RAGAgent instance.
        
        Args:
            agent: RAGAgent instance to wrap
        """
        self.agent = agent
        self._original_agent_loop = agent._agent_loop
        
        # Wrap the agent loop
        agent._agent_loop = self._traced_agent_loop
    
    def _traced_agent_loop(self, standalone_query: str):
        """Traced version of agent loop."""
        global _tracer
        
        if _tracer is None:
            return self._original_agent_loop(standalone_query)
        
        with _tracer.start_as_current_span("agent_loop") as span:
            span.set_attribute("query", standalone_query)
            
            result, sources = self._original_agent_loop(standalone_query)
            
            span.set_attribute("num_sources", len(sources))
            span.set_attribute("response_length", len(result) if result else 0)
            
            return result, sources


def main():
    """Test Phoenix integration."""
    # Initialize Phoenix
    if not initialize_phoenix(launch_app=True):
        print("Failed to initialize Phoenix")
        return
    
    # Test tracing
    with trace_span("test_span", {"test_key": "test_value"}):
        print("Inside traced span")
        
        # Simulate search trace
        trace_search(
            query="test query",
            results=[{"score": 0.9, "content": "test"}],
            source="hybrid",
        )
    
    print("Test complete. Check Phoenix UI at http://localhost:6006")
    
    # Keep app running
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

"""
Agentic RAG flow with conversation history and streaming.

Implements:
1. Query rewriting (conversation → standalone query)
2. RAG agent with ReAct-style reasoning (no tool_calls, more reliable)
3. Streaming thoughts and responses
4. Iterative reasoning with max iterations
"""

import json
import os
import re
from typing import Optional, Generator
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings
from src.llm.client import LLMClient
from src.indexing.hybrid_search import HybridSearch

# Tracing imports - use OpenTelemetry tracer (configured by Phoenix)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    tracer = trace.get_tracer(__name__)
    TRACING_ENABLED = True
except ImportError:
    tracer = None
    TRACING_ENABLED = False
    print("OpenTelemetry not available - tracing disabled")


@dataclass
class AgentState:
    """State for the RAG agent."""
    
    conversation_history: list[dict] = field(default_factory=list)
    current_query: str = ""
    standalone_query: str = ""
    search_results: list[dict] = field(default_factory=list)
    iterations: int = 0
    final_answer: Optional[str] = None


class RAGAgent:
    """Agentic RAG with query rewriting and ReAct-style reasoning."""
    
    # System prompt for query rewriting
    QUERY_REWRITE_PROMPT = """You are a query rewriting assistant. Your task is to rewrite the user's latest message into a standalone question that can be understood without the conversation history.

Rules:
1. If the latest message references previous context (like "it", "that", "this"), incorporate that context
2. Keep the rewritten question concise and clear
3. Don't add information not present in the conversation
4. If the latest message is already standalone, return it as-is
5. Output ONLY the rewritten question, nothing else

Conversation history:
{history}

Latest user message: {query}

Rewritten standalone question:"""

    # System prompt for RAG agent with ReAct-style reasoning
    RAG_SYSTEM_PROMPT = """You are a helpful Robinhood support assistant. You help users understand Robinhood's products, services, and features.

You must follow this exact format for your responses:

<thinking>
Your step-by-step reasoning about what information you need and how to answer the question.
</thinking>

<search>query text here</search>

OR if you have enough information:

<thinking>
Your reasoning about why you can now answer the question.
</thinking>

<answer>
Your final answer to the user's question. Be direct and helpful. Cite sources using [1], [2], etc.
</answer>

IMPORTANT RULES:
1. Always start with <thinking> tags to show your reasoning
2. Use <search>query</search> when you need information from the documentation
3. Use <answer>response</answer> when you have enough information to answer
4. Never output raw function calls or code - only use the XML tags above
5. Be concise but thorough in your answers
6. Always cite sources when providing information from search results"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        hybrid_search: Optional[HybridSearch] = None,
        max_iterations: Optional[int] = None,
    ):
        """Initialize the RAG agent.
        
        Args:
            llm_client: LLMClient instance
            hybrid_search: HybridSearch instance
            max_iterations: Maximum agent iterations
        """
        self.llm = llm_client or LLMClient()
        self.search = hybrid_search or HybridSearch()
        self.max_iterations = max_iterations or settings.MAX_AGENT_ITERATIONS
        
        # Conversation history (persists across calls)
        self.conversation_history: list[dict] = []
    
    def _rewrite_query(self, query: str) -> str:
        """Rewrite query to standalone question using conversation history."""
        # Add tracing for query rewrite step
        if TRACING_ENABLED and tracer:
            with tracer.start_as_current_span("query_rewrite") as span:
                span.set_attribute("input.value", query)
                span.set_attribute("openinference.span.kind", "CHAIN")
                result = self._rewrite_query_internal(query)
                span.set_attribute("output.value", result)
                return result
        else:
            return self._rewrite_query_internal(query)
    
    def _rewrite_query_internal(self, query: str) -> str:
        """Internal query rewrite logic."""
        if len(self.conversation_history) < 2:
            return query
        
        history_str = ""
        for msg in self.conversation_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
        
        prompt = self.QUERY_REWRITE_PROMPT.format(
            history=history_str.strip(),
            query=query,
        )
        
        rewritten = self.llm.simple_completion(
            prompt=prompt,
            temperature=0.3,
        )
        
        return rewritten.strip()
    
    def _execute_search(self, query: str) -> list[dict]:
        """Execute hybrid search."""
        # Add tracing for search tool
        if TRACING_ENABLED and tracer:
            with tracer.start_as_current_span("hybrid_search") as span:
                span.set_attribute("input.value", query)
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", "hybrid_search")
                span.set_attribute("tool.description", "Search Robinhood support documentation")
                results = self.search.search(
                    query=query,
                    k=settings.SEARCH_TOP_K,
                )
                span.set_attribute("output.value", f"Found {len(results)} results")
                return results
        else:
            results = self.search.search(
                query=query,
                k=settings.SEARCH_TOP_K,
            )
            return results
    
    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results for the LLM."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("metadata", {}).get("title", "Unknown")
            url = r.get("metadata", {}).get("source_url", "")
            content = r.get("content", "")[:500]
            
            formatted.append(
                f"[{i}] {title}\n"
                f"Source: {url}\n"
                f"Content: {content}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def _parse_response(self, response: str) -> dict:
        """Parse the agent's response to extract thinking, search, and answer.
        
        Returns:
            Dict with 'thinking', 'search_query', 'answer', and 'raw'
        """
        result = {
            "thinking": None,
            "search_query": None,
            "answer": None,
            "raw": response,
        }
        
        # Extract thinking
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        if thinking_match:
            result["thinking"] = thinking_match.group(1).strip()
        
        # Extract search query
        search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
        if search_match:
            result["search_query"] = search_match.group(1).strip()
        
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        return result
    
    def _agent_loop(self, standalone_query: str) -> tuple[str, list[dict], list[str]]:
        """Run the agent loop with ReAct-style reasoning.
        
        Args:
            standalone_query: The standalone query to process
            
        Returns:
            Tuple of (final answer, search results used, list of thoughts)
        """
        messages = [
            {"role": "system", "content": self.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": standalone_query},
        ]
        
        all_search_results = []
        all_thoughts = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Force final answer on last iteration
            if iterations == self.max_iterations:
                messages.append({
                    "role": "system",
                    "content": "You must provide your final answer now using <answer></answer> tags. Do not search again."
                })
            
            # Get LLM response
            response = self.llm.chat(messages, temperature=0.7)
            response_text = response.get("content", "")
            
            # Parse the response
            parsed = self._parse_response(response_text)
            
            # Collect thinking
            if parsed["thinking"]:
                all_thoughts.append(parsed["thinking"])
            
            # If there's a final answer, we're done
            if parsed["answer"]:
                return parsed["answer"], all_search_results, all_thoughts
            
            # If there's a search query, execute it
            if parsed["search_query"]:
                results = self._execute_search(parsed["search_query"])
                all_search_results.extend(results)
                
                # Format results and add to messages
                result_text = self._format_search_results(results)
                
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Search results for '{parsed['search_query']}':\n\n{result_text}\n\nNow provide your answer using <answer></answer> tags, or search again if needed."
                })
            else:
                # No search or answer - try to extract answer from raw response
                if response_text.strip():
                    return response_text, all_search_results, all_thoughts
        
        return "I apologize, but I couldn't complete the search.", all_search_results, all_thoughts
    
    def chat(self, user_message: str) -> dict:
        """Process a user message and return a response.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with 'response', 'standalone_query', 'sources', and 'thoughts'
        """
        # Create parent span for entire RAG request
        if TRACING_ENABLED and tracer:
            with tracer.start_as_current_span("rag_request") as span:
                span.set_attribute("input.value", user_message)
                span.set_attribute("openinference.span.kind", "AGENT")
                result = self._chat_internal(user_message)
                span.set_attribute("output.value", result.get("response", ""))
                span.set_status(Status(StatusCode.OK))
                return result
        else:
            return self._chat_internal(user_message)
    
    def _chat_internal(self, user_message: str) -> dict:
        """Internal chat processing (wrapped by tracing in chat())."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Rewrite query to standalone
        standalone_query = self._rewrite_query(user_message)
        
        # Run agent loop
        answer, search_results, thoughts = self._agent_loop(standalone_query)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": answer,
        })
        
        # Extract unique sources
        sources = []
        seen_urls = set()
        for r in search_results:
            url = r.get("metadata", {}).get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url": url,
                    "title": r.get("metadata", {}).get("title", ""),
                })
        
        return {
            "response": answer,
            "standalone_query": standalone_query,
            "sources": sources,
            "thoughts": thoughts,
        }
    
    def chat_stream(self, user_message: str) -> Generator[dict, None, None]:
        """Process a user message with streaming.
        
        Yields events for:
        - thoughts: Thinking process chunks
        - search: Search being performed
        - results: Search results
        - answer: Final answer chunks
        
        Args:
            user_message: The user's message
            
        Yields:
            Dict with 'type' and 'content'
        """
        # Create parent span for entire RAG request (wraps entire generator)
        if TRACING_ENABLED and tracer:
            with tracer.start_as_current_span("rag_request") as span:
                span.set_attribute("input.value", user_message)
                span.set_attribute("openinference.span.kind", "AGENT")
                
                # Yield all events from internal generator
                for event in self._chat_stream_internal(user_message):
                    yield event
                
                # Set output after completion
                if event.get("type") == "complete":
                    span.set_attribute("output.value", event.get("content", ""))
                span.set_status(Status(StatusCode.OK))
        else:
            yield from self._chat_stream_internal(user_message)
    
    def _chat_stream_internal(self, user_message: str) -> Generator[dict, None, None]:
        """Internal streaming implementation."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Rewrite query
        standalone_query = self._rewrite_query(user_message)
        yield {"type": "status", "content": f"Searching: {standalone_query}"}
        
        # Build messages
        messages = [
            {"role": "system", "content": self.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": standalone_query},
        ]
        
        all_search_results = []
        all_thoughts = []
        iterations = 0
        final_answer = ""
        
        while iterations < self.max_iterations:
            iterations += 1
            
            if iterations == self.max_iterations:
                messages.append({
                    "role": "system",
                    "content": "You must provide your final answer now. Do not search again."
                })
            
            # Stream the response
            full_response = ""
            current_buffer = ""
            in_thinking = False
            in_search = False
            in_answer = False
            
            for chunk in self.llm.chat_stream(messages, temperature=0.7):
                full_response += chunk
                current_buffer += chunk
                
                # Check for tag starts
                if "<thinking>" in current_buffer and not in_thinking:
                    in_thinking = True
                    yield {"type": "thinking_start", "content": ""}
                    current_buffer = current_buffer.split("<thinking>")[-1]
                
                if in_thinking and "</thinking>" not in full_response:
                    yield {"type": "thinking", "content": chunk}
                
                if "</thinking>" in current_buffer and in_thinking:
                    in_thinking = False
                    yield {"type": "thinking_end", "content": ""}
                    current_buffer = current_buffer.split("</thinking>")[-1]
                
                if "<search>" in current_buffer and not in_search:
                    in_search = True
                
                if "<answer>" in current_buffer and not in_answer:
                    in_answer = True
                    yield {"type": "answer_start", "content": ""}
                    current_buffer = current_buffer.split("<answer>")[-1]
                
                if in_answer and "</answer>" not in full_response:
                    yield {"type": "answer", "content": chunk}
            
            # Parse the complete response
            parsed = self._parse_response(full_response)
            
            if parsed["thinking"]:
                all_thoughts.append(parsed["thinking"])
            
            # Handle final answer
            if parsed["answer"]:
                final_answer = parsed["answer"]
                yield {"type": "answer_end", "content": ""}
                break
            
            # Handle search
            if parsed["search_query"]:
                yield {"type": "search", "content": parsed["search_query"]}
                
                results = self._execute_search(parsed["search_query"])
                all_search_results.extend(results)
                
                yield {"type": "results", "content": f"Found {len(results)} results"}
                
                result_text = self._format_search_results(results)
                
                messages.append({"role": "assistant", "content": full_response})
                messages.append({
                    "role": "user",
                    "content": f"Search results:\n\n{result_text}\n\nNow provide your answer."
                })
            else:
                # No structured output - use raw
                final_answer = full_response
                break
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_answer,
        })
        
        # Extract sources
        sources = []
        seen_urls = set()
        for r in all_search_results:
            url = r.get("metadata", {}).get("source_url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "url": url,
                    "title": r.get("metadata", {}).get("title", ""),
                })
        
        yield {
            "type": "complete",
            "content": final_answer,
            "sources": sources,
            "thoughts": all_thoughts,
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


def main():
    """Test the RAG agent."""
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Test conversation
    test_messages = [
        "What is Robinhood?",
        "How do I trade options on it?",
    ]
    
    for msg in test_messages:
        print(f"\n{'=' * 50}")
        print(f"User: {msg}")
        print("=" * 50)
        
        result = agent.chat(msg)
        
        print(f"\nStandalone Query: {result['standalone_query']}")
        
        if result.get('thoughts'):
            print(f"\nThoughts:")
            for i, thought in enumerate(result['thoughts'], 1):
                print(f"  {i}. {thought[:100]}...")
        
        print(f"\nResponse: {result['response']}")
        
        if result['sources']:
            print(f"\nSources:")
            for src in result['sources']:
                print(f"  - {src['title']}: {src['url']}")


if __name__ == "__main__":
    main()

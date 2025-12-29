"""
OpenAI-compatible LLM client.

Supports any OpenAI-compatible API including DeepSeek, local models, etc.
"""

import os
from typing import Optional, Generator

from openai import OpenAI

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import settings


class LLMClient:
    """OpenAI-compatible LLM client."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the LLM client.
        
        Args:
            base_url: API base URL (default from settings)
            api_key: API key (default from settings)
            model: Model name (default from settings)
        """
        self.base_url = base_url or settings.LLM_BASE_URL
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or settings.LLM_MODEL
        
        if not self.api_key:
            raise ValueError(
                "LLM API key not set. "
                "Set LLM_API_KEY in your .env file or pass it directly."
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        print(f"LLM client initialized: {self.base_url} / {self.model}")
    
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice mode ('auto', 'none', or specific)
            
        Returns:
            Response dict with 'content', 'role', and optionally 'tool_calls'
        """
        # Build request params
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        if tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice
        
        # Make request
        response = self.client.chat.completions.create(**params)
        
        # Extract response
        choice = response.choices[0]
        message = choice.message
        
        result = {
            "role": "assistant",
            "content": message.content,
        }
        
        # Include tool calls if present
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result
    
    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Stream a chat completion response.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Response content chunks
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**params)
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Simple completion with optional system prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Returns:
            Response content string
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        result = self.chat(messages, temperature=temperature)
        return result["content"] or ""


def main():
    """Test LLM client."""
    client = LLMClient()
    
    # Simple test
    response = client.simple_completion(
        prompt="What is Robinhood?",
        system_prompt="You are a helpful assistant. Keep responses brief.",
    )
    
    print("Response:", response)


if __name__ == "__main__":
    main()

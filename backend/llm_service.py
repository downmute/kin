"""
LLM Service - Integration with JanitorAI for text generation
Extracted and simplified from the original Flask app.
"""
import os
import json
import logging
from typing import AsyncIterator, Optional
import httpx


class LLMService:
    """Service for interacting with LLM API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM service.
        
        Args:
            api_key: API key
            model: Model to use (default: "gpt-4o-mini")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        logging.info(f"LLM Service initialized with model: {self.model}")
    
    def _build_system_prompt(self, persona: Optional[str] = None, backstory: Optional[str] = None) -> str:
        """
        Build system prompt with persona and backstory.
        
        Args:
            persona: Optional persona/character description
            backstory: Optional backstory
        
        Returns:
            System prompt string
        """
        system_prompt = "You are a helpful AI assistant. Keep responses concise and conversational (2-3 sentences max)."
        
        if persona:
            system_prompt = f"{persona} Keep responses short and natural (2-3 sentences max)."
        
        if backstory:
            system_prompt += f" Background: {backstory}"
        
        return system_prompt
    
    async def stream_response(
        self,
        text: str,
        persona: Optional[str] = None,
        backstory: Optional[str] = None,
        max_tokens: int = 150
    ) -> AsyncIterator[str]:
        """
        Stream LLM response as text chunks.
        
        Args:
            text: User input text
            persona: Optional persona/character description
            backstory: Optional backstory
            max_tokens: Maximum tokens in response
        
        Yields:
            Text chunks from the LLM stream
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        system_prompt = self._build_system_prompt(persona, backstory)
        
        llm_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "stream": True,
            "max_tokens": max_tokens
        }
        
        logging.info(f"Sending LLM request: '{text[:50]}...'")
        
        buffer = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.api_url, headers=headers, json=llm_data) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8')
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        
                        if line.startswith("data:"):
                            json_str = line[len("data:"):].strip()
                            if not json_str:
                                continue
                            
                            try:
                                json_data = json.loads(json_str)
                                delta_content = json_data.get("choices", [{}])[0].get("delta", {}).get("content")
                                
                                if delta_content:
                                    yield delta_content
                                    
                            except json.JSONDecodeError:
                                logging.warning(f"Could not decode JSON from line: {json_str}")
    
    async def get_full_response(
        self,
        text: str,
        persona: Optional[str] = None,
        backstory: Optional[str] = None,
        max_tokens: int = 150
    ) -> str:
        """
        Get complete LLM response as a single string.
        
        Args:
            text: User input text
            persona: Optional persona/character description
            backstory: Optional backstory
            max_tokens: Maximum tokens in response
        
        Returns:
            Complete response text
        """
        full_response = ""
        async for chunk in self.stream_response(text, persona, backstory, max_tokens):
            full_response += chunk
        return full_response


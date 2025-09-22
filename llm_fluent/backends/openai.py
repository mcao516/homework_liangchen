"""Backend implementation for OpenAI."""

import logging
from typing import Optional

from openai import OpenAI, AsyncOpenAI
from llm_fluent.backends.base import LLMBackend

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI API compatible backend."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """Initialize OpenAI backend.
        
        Args:
            api_key: API key for authentication.
            model: Model name to use.
            base_url: Optional base URL for API (for OpenAI-compatible services).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Run: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate response using OpenAI API."""
        try:
            response = await self.async_client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI async generation failed: {e}")
            raise
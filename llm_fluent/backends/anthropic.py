"""Backend implementation for Anthropic."""

import logging

from llm_fluent.backends.base import Backend

logger = logging.getLogger(__name__)


class AnthropicBackend(Backend):
    """Anthropic API backend."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """Initialize Anthropic backend.
        
        Args:
            api_key: API key for authentication.
            model: Model name to use.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate response using Anthropic API."""
        try:
            response = await self.async_client.messages.create(
                model=kwargs.get("model", self.model),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as e:
            logger.error(f"Anthropic async generation failed: {e}")
            raise

"""Abstract base classes for LLM backends."""

import abc
import asyncio

from typing import Optional, List


class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends."""
    
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response from the LLM."""
        pass
    
    @abc.abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate a single response from the LLM."""
        pass


class MockBackend(LLMBackend):
    """Mock backend for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None, cycle: bool = True):
        """Initialize mock backend with predefined responses.
        
        Args:
            responses: List of responses to return.
            cycle: If True, cycle through responses infinitely. 
                   If False, raise StopIteration after all responses are consumed.
        """
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.cycle = cycle
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        if not self.cycle and self.call_count >= len(self.responses):
            raise StopIteration("No more mock responses available")
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Return a mock response asynchronously."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self.generate(prompt, **kwargs)
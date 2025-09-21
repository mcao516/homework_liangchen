"""Core components for the fluent LLM interface."""
import asyncio
import logging
from typing import (
    Callable, Generic, Iterator, Optional, Type, TypeVar,
    AsyncIterator, List
)

from llm_fluent.backends.base import Backend
from llm_fluent.extract_utils import extract_from_text

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


# ============================================================================
# Fluent Chain Classes
# ============================================================================

class Chain(Generic[T]):
    """Base class for fluent chains."""
    
    def __init__(self, source: Iterator[T]):
        """Initialize chain with a source iterator.
        
        Args:
            source: Iterator that provides values for the chain.
        """
        self._source = source
    
    def __iter__(self) -> Iterator[T]:
        """Make the chain iterable."""
        return self._source
    
    def extract(self, schema: Type[U]) -> 'Chain[U]':
        """Extract structured data from text responses.
        
        Args:
            schema: Target schema for extraction.
            
        Returns:
            New chain with extracted objects.
        """
        return self.map(lambda text: extract_from_text(str(text), schema))
    
    def map(self, func: Callable[[T], U]) -> 'Chain[U]':
        """Transform each item using the given function.
        
        Args:
            func: Function to apply to each item.
            
        Returns:
            New chain with transformed items.
        """
        return Chain(func(item) for item in self._source)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Chain[T]':
        """Filter items based on predicate.
        
        Args:
            predicate: Function that returns True for items to keep.
            
        Returns:
            New chain with filtered items.
        """
        return Chain(item for item in self._source if predicate(item))
    
    def take(self, n: int) -> List[T]:
        """Take first n items from the chain.
        
        Args:
            n: Number of items to take.
            
        Returns:
            List of first n items.
        """
        result = []
        for i, item in enumerate(self._source):
            if i >= n:
                break
            result.append(item)
        return result
    
    def collect(self) -> List[T]:
        """Collect all items into a list.
        
        Returns:
            List of all items in the chain.
        """
        return list(self._source)


class AsyncChain(Generic[T]):
    """Async version of Chain."""
    
    def __init__(self, source: AsyncIterator[T]):
        """Initialize async chain with an async source iterator."""
        self._source = source
    
    def __aiter__(self) -> AsyncIterator[T]:
        """Make the chain async iterable."""
        return self._source
    
    def extract(self, schema: Type[U]) -> 'AsyncChain[U]':
        """Extract structured data from text responses.
        
        Args:
            schema: Target schema for extraction.
            
        Returns:
            New async chain with extracted objects.
        """
        return self.map(lambda text: extract_from_text(str(text), schema))
    
    def map(self, func: Callable[[T], U]) -> 'AsyncChain[U]':
        """Transform each item using the given function."""
        async def _map():
            async for item in self._source:
                yield func(item)
        return AsyncChain(_map())
    
    def filter(self, predicate: Callable[[T], bool]) -> 'AsyncChain[T]':
        """Filter items based on predicate."""
        async def _filter():
            async for item in self._source:
                if predicate(item):
                    yield item
        return AsyncChain(_filter())
    
    async def take(self, n: int) -> List[T]:
        """Take first n items from the chain."""
        result = []
        i = 0
        async for item in self._source:
            if i >= n:
                break
            result.append(item)
            i += 1
        return result
    
    async def collect(self) -> List[T]:
        """Collect all items into a list."""
        return [item async for item in self._source]


# ============================================================================
# Main Prompt Classes
# ============================================================================

class Prompt:
    """Synchronous fluent prompt builder."""
    
    def __init__(
        self,
        prompt: str,
        backend: Backend,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_iterations: Optional[int] = None
    ):
        """Initialize prompt with backend.
        
        Args:
            prompt: The prompt text.
            backend: Backend to use for generation.
            max_retries: Maximum retries on failure.
            retry_delay: Delay between retries in seconds.
            max_iterations: Maximum number of iterations (None for infinite).
        """
        self.prompt = prompt
        self.backend = backend
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_iterations = max_iterations
    
    def sample(self) -> Chain[str]:
        """Create a stream of LLM responses.
        
        Returns:
            Chain of response strings.
        """
        def _generate():
            iteration = 0
            while self.max_iterations is None or iteration < self.max_iterations:
                for attempt in range(self.max_retries):
                    try:
                        response = self.backend.generate(self.prompt)
                        yield response
                        iteration += 1
                        break
                    except StopIteration:
                        # Backend has no more responses
                        return
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(f"Max retries exceeded: {e}")
                            raise
                        logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                        import time
                        time.sleep(self.retry_delay)
        
        return Chain(_generate())


class AsyncPrompt:
    """Asynchronous fluent prompt builder."""
    
    def __init__(
        self,
        prompt: str,
        backend: Backend,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_iterations: Optional[int] = None
    ):
        """Initialize async prompt with backend.
        
        Args:
            prompt: The prompt text.
            backend: Backend to use for generation.
            max_retries: Maximum retries on failure.
            retry_delay: Delay between retries in seconds.
            max_iterations: Maximum number of iterations (None for infinite).
        """
        self.prompt = prompt
        self.backend = backend
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_iterations = max_iterations
    
    def sample(self) -> AsyncChain[str]:
        """Create a stream of LLM responses."""
        async def _generate():
            iteration = 0
            while self.max_iterations is None or iteration < self.max_iterations:
                for attempt in range(self.max_retries):
                    try:
                        response = await self.backend.agenerate(self.prompt)
                        yield response
                        iteration += 1
                        break
                    except StopIteration:
                        # Backend has no more responses
                        return
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(f"Max retries exceeded: {e}")
                            raise
                        logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                        await asyncio.sleep(self.retry_delay)
        
        return AsyncChain(_generate())
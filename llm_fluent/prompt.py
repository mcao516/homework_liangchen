"""Core components for the fluent LLM interface."""
import asyncio
import logging
import time
import inspect
from dataclasses import is_dataclass
from typing import (
    Any, AsyncIterator, Callable, Generic, Iterator, Optional, 
    Type, TypeVar, Union, Dict, List
)

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    ValidationError = None

from llm_fluent.backends.base import LLMBackend
from llm_fluent.extractor import SchemaExtractor

logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


class FluentChain(Generic[T]):
    """Base class for fluent chain operations."""
    
    def __init__(self, source: Iterator[T]):
        """Initialize the fluent chain.
        
        Args:
            source: Source iterator for the chain.
        """
        self._source = source
    
    def __iter__(self) -> Iterator[T]:
        """Return iterator for the chain."""
        return self._source
    
    def map(self, func: Callable[[T], U]) -> 'FluentChain[U]':
        """Transform each item using the provided function.
        
        Args:
            func: Transformation function.
            
        Returns:
            New FluentChain with transformed items.
        """
        def generator():
            for item in self._source:
                yield func(item)
        return FluentChain(generator())
    
    def filter(self, predicate: Callable[[T], bool]) -> 'FluentChain[T]':
        """Filter items based on predicate.
        
        Args:
            predicate: Filter function.
            
        Returns:
            New FluentChain with filtered items.
        """
        def generator():
            for item in self._source:
                if predicate(item):
                    yield item
        return FluentChain(generator())
    
    def take(self, n: int) -> List[T]:
        """Take first n items from the chain.
        
        Args:
            n: Number of items to take.
            
        Returns:
            List of up to n items (may be fewer if stream has fewer items).
        """
        result = []
        for item in self._source:
            if len(result) >= n:
                break
            result.append(item)
        return result
    
    def extract(
        self,
        schema: Union[Type[U], Dict[str, Any]]
    ) -> 'FluentChain[U]':
        """Extract structured data from responses.
        
        Args:
            schema: Target schema for extraction.
            
        Returns:
            New FluentChain with extracted objects.
        """
        def generator():
            for item in self._source:
                if not isinstance(item, str):
                    item = str(item)
                
                # Extract all JSON objects from the response
                json_objects = SchemaExtractor.extract_all_json_from_text(item)
                
                for json_data in json_objects:
                    try:
                        if isinstance(schema, dict):
                            yield json_data
                        elif is_dataclass(schema):
                            yield SchemaExtractor.parse_to_dataclass(json_data, schema)
                        elif HAS_PYDANTIC and inspect.isclass(schema) and issubclass(schema, BaseModel):
                            yield SchemaExtractor.parse_to_pydantic(json_data, schema)
                        else:
                            yield json_data
                    except (ValueError, TypeError, ValidationError if HAS_PYDANTIC else Exception):
                        logger.info("Object cannot be created using schema:", json_data)
                        continue
        
        return FluentChain(generator())
    
    def collect(self) -> List[T]:
        """Collect all items into a list.
        
        Returns:
            List of all items in the chain.
        
        Raises:
            RuntimeError: If attempting to collect from an infinite stream.
        """
        return list(self._source)


class AsyncFluentChain(Generic[T]):
    """Async version of FluentChain."""
    
    def __init__(self, source: AsyncIterator[T]):
        """Initialize the async fluent chain.
        
        Args:
            source: Source async iterator for the chain.
        """
        self._source = source
    
    def __aiter__(self) -> AsyncIterator[T]:
        """Return async iterator for the chain."""
        return self._source
    
    def map(self, func: Callable[[T], U]) -> 'AsyncFluentChain[U]':
        """Transform each item using the provided function.
        
        Args:
            func: Transformation function.
            
        Returns:
            New AsyncFluentChain with transformed items.
        """
        async def generator():
            async for item in self._source:
                if asyncio.iscoroutinefunction(func):
                    yield await func(item)
                else:
                    yield func(item)
        return AsyncFluentChain(generator())
    
    def filter(self, predicate: Callable[[T], bool]) -> 'AsyncFluentChain[T]':
        """Filter items based on predicate.
        
        Args:
            predicate: Filter function.
            
        Returns:
            New AsyncFluentChain with filtered items.
        """
        async def generator():
            async for item in self._source:
                if asyncio.iscoroutinefunction(predicate):
                    if await predicate(item):
                        yield item
                else:
                    if predicate(item):
                        yield item
        return AsyncFluentChain(generator())
    
    async def take(self, n: int) -> List[T]:
        """Take first n items from the chain.
        
        Args:
            n: Number of items to take.
            
        Returns:
            List of up to n items (may be fewer if stream has fewer items).
        """
        result = []
        async for item in self._source:
            if len(result) >= n:
                break
            result.append(item)
        return result
    
    def extract(
        self,
        schema: Union[Type[U], Dict[str, Any]]
    ) -> 'AsyncFluentChain[U]':
        """Extract structured data from responses.
        
        Args:
            schema: Target schema for extraction.
            
        Returns:
            New AsyncFluentChain with extracted objects.
        """
        async def generator():
            async for item in self._source:
                if not isinstance(item, str):
                    item = str(item)
                
                # Extract all JSON objects from the response
                json_objects = SchemaExtractor.extract_all_json_from_text(item)
                
                for json_data in json_objects:
                    try:
                        if isinstance(schema, dict):
                            yield json_data
                        elif is_dataclass(schema):
                            yield SchemaExtractor.parse_to_dataclass(json_data, schema)
                        elif HAS_PYDANTIC and inspect.isclass(schema) and issubclass(schema, BaseModel):
                            yield SchemaExtractor.parse_to_pydantic(json_data, schema)
                        else:
                            yield json_data
                    except (ValueError, TypeError, ValidationError if HAS_PYDANTIC else Exception):
                        continue
        
        return AsyncFluentChain(generator())
    
    async def collect(self) -> List[T]:
        """Collect all items into a list.
        
        Returns:
            List of all items in the chain.
        
        Raises:
            RuntimeError: If attempting to collect from an infinite stream.
        """
        result = []
        async for item in self._source:
            result.append(item)
        return result


class Prompt:
    """Main entry point for fluent LLM prompting."""
    
    def __init__(
        self,
        prompt: str,
        backend: LLMBackend,
        extraction_schema: Optional[Union[Type, Dict[str, Any]]] = None
    ):
        """Initialize a Prompt.
        
        Args:
            prompt: The prompt string.
            backend: LLM backend to use.
            extraction_schema: Optional schema for automatic extraction.
        """
        self.prompt = prompt
        self.backend = backend
        self.extraction_schema = extraction_schema
        
        if extraction_schema:
            self.prompt = SchemaExtractor.create_extraction_prompt(
                prompt, extraction_schema
            )
    
    def sample(
        self, 
        max_iterations: Optional[int] = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> FluentChain[str]:
        """Generate responses from the LLM with retry logic.
        
        Args:
            max_iterations: Maximum number of API calls to make (default=1).
                           Set to None for infinite generation (use with caution).
            max_retries: Maximum number of retry attempts per API call (default=3).
            retry_delay: Delay in seconds between retry attempts (default=1.0).
                        Uses exponential backoff: delay * (2 ** attempt).
            
        Returns:
            FluentChain of responses.
            
        Raises:
            Exception: If all retry attempts fail for a single API call.
        """
        def generator():
            iteration = 0
            while max_iterations is None or iteration < max_iterations:
                response = None
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        response = self.backend.generate(self.prompt)
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            # Exponential backoff
                            sleep_time = retry_delay * (2 ** attempt)
                            time.sleep(sleep_time)
                        else:
                            # All retries exhausted
                            error_msg = (
                                f"API call failed after {max_retries + 1} attempts. "
                                f"Last error: {last_error}"
                            )
                            raise Exception(error_msg) from last_error
                
                if response is not None:
                    yield response
                    iteration += 1
                
                # If we've reached the limit, stop generating
                if max_iterations is not None and iteration >= max_iterations:
                    break
        
        return FluentChain(generator())
    
    def asample(
            self,
            max_iterations: Optional[int] = 1,
            max_retries: int = 3,
            retry_delay: float = 1.0  
        ) -> AsyncFluentChain[str]:
        """Asynchronously generate responses in a loop.
        
        Args:
            max_iterations: Maximum number of iterations (None for infinite).
            max_retries: Maximum number of retry attempts per API call (default=3).
            retry_delay: Delay in seconds between retry attempts (default=1.0).
                        Uses exponential backoff: delay * (2 ** attempt).
        
        Returns:
            AsyncFluentChain of responses.
        """
        async def generator():
            iteration = 0
        
            while max_iterations is None or iteration < max_iterations:
                response = None
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        response = await self.backend.agenerate(self.prompt)
                        break  # Success, exit retry loop
                    except Exception as e:
                        last_error = e
                        if attempt < max_retries:
                            # Exponential backoff
                            sleep_time = retry_delay * (2 ** attempt)
                            time.sleep(sleep_time)
                        else:
                            # All retries exhausted
                            error_msg = (
                                f"API call failed after {max_retries + 1} attempts. "
                                f"Last error: {last_error}"
                            )
                            raise Exception(error_msg) from last_error
                
                if response is not None:
                    yield response
                    iteration += 1
                
                # If we've reached the limit, stop generating
                if max_iterations is not None and iteration >= max_iterations:
                    break

        return AsyncFluentChain(generator())
    
    def generate(self) -> str:
        """Generate a single response.
        
        Returns:
            Single response string.
        """
        return self.backend.generate(self.prompt)
    
    async def agenerate(self) -> str:
        """Asynchronously generate a single response.
        
        Returns:
            Single response string.
        """
        return await self.backend.agenerate(self.prompt)
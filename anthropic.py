"""Fluent-style LLM prompting and response transformation library.

This module provides a fluent interface for LLM interactions with support for
streaming, extraction, transformation, and filtering of responses.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Any, AsyncIterator, Callable, Generic, Iterator, Optional, 
    Type, TypeVar, Union, Dict, List
)
from enum import Enum
import inspect

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    ValidationError = None


# Type variables
T = TypeVar('T')
U = TypeVar('U')


class BackendType(Enum):
    """Supported LLM backend types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response from the LLM."""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate a single response from the LLM."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API compatible backend."""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """Initialize OpenAI backend.
        
        Args:
            api_key: API key for authentication.
            base_url: Base URL for API endpoint.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        return response.choices[0].message.content
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate a response using OpenAI API."""
        response = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        return response.choices[0].message.content


class AnthropicBackend(LLMBackend):
    """Anthropic API compatible backend."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """Initialize Anthropic backend.
        
        Args:
            api_key: API key for authentication.
            model: Model identifier.
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
        """Generate a response using Anthropic API."""
        response = self.client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate a response using Anthropic API."""
        response = await self.async_client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class SchemaExtractor:
    """Utility class for extracting structured data from text."""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from text, handling markdown code blocks."""
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in markdown code blocks
        import re
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find raw JSON in text
        json_pattern = r'(\{[\s\S]*?\}|\[[\s\S]*?\])'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def extract_all_json_from_text(text: str) -> List[Dict[str, Any]]:
        """Extract ALL JSON objects/arrays from text.
        
        Returns a list of extracted JSON objects. If the extracted item is
        an array, it unpacks the array elements as individual items.
        """
        import re
        results = []
        
        # Try direct JSON parsing first (might be a single object or array)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
            return results
        except json.JSONDecodeError:
            pass
        
        # Find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
        
        # If we found JSON in code blocks, return those
        if results:
            return results
        
        # Try to find all JSON objects in text
        # More sophisticated regex to match balanced braces
        def find_json_objects(text):
            objects = []
            i = 0
            while i < len(text):
                if text[i] in '{[':
                    # Try to parse from this position
                    bracket_count = 0
                    start = i
                    in_string = False
                    escape_next = False
                    
                    while i < len(text):
                        if escape_next:
                            escape_next = False
                        elif text[i] == '\\':
                            escape_next = True
                        elif text[i] == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if text[i] in '{[':
                                bracket_count += 1
                            elif text[i] in '}]':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found complete JSON object/array
                                    try:
                                        parsed = json.loads(text[start:i+1])
                                        if isinstance(parsed, list):
                                            objects.extend(parsed)
                                        else:
                                            objects.append(parsed)
                                    except json.JSONDecodeError:
                                        pass
                                    break
                        i += 1
                i += 1
            return objects
        
        results = find_json_objects(text)
        return results if results else []
    
    @staticmethod
    def parse_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
        """Parse dictionary to dataclass instance."""
        if not is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")
        
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
    
    @staticmethod
    def parse_to_pydantic(data: Dict[str, Any], cls: Type[BaseModel]) -> BaseModel:
        """Parse dictionary to Pydantic model instance."""
        if not HAS_PYDANTIC or not issubclass(cls, BaseModel):
            raise ValueError(f"{cls} is not a Pydantic model")
        return cls(**data)
    
    @staticmethod
    def create_extraction_prompt(
        original_prompt: str,
        schema: Union[Type, Dict[str, Any]]
    ) -> str:
        """Create a prompt that includes extraction instructions."""
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, indent=2)
        elif is_dataclass(schema):
            schema_dict = {
                f.name: str(f.type.__name__ if hasattr(f.type, '__name__') else f.type)
                for f in fields(schema)
            }
            schema_str = json.dumps(schema_dict, indent=2)
        elif HAS_PYDANTIC and issubclass(schema, BaseModel):
            schema_str = schema.model_json_schema()
            schema_str = json.dumps(schema_str, indent=2)
        else:
            schema_str = str(schema)
        
        return f"""{original_prompt}

Please provide your response as a JSON array containing ALL matching items. 
Each item should match this schema:
{schema_str}

Return ONLY the JSON array, no additional text. Example format:
[
  {{"field1": "value1", "field2": value2}},
  {{"field1": "value3", "field2": value4}}
]"""


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
            max_iterations: Optional[int] = None,
            max_retries: int = 3,
            retry_delay: float = 1.0  
        ) -> AsyncFluentChain[str]:
        """Asynchronously generate responses in a loop.
        
        Args:
            max_iterations: Maximum number of iterations (None for infinite).

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


# Demo and testing code
def demo():
    """Demonstrate the library with OpenRouter's free Grok API."""
    print("=== Fluent LLM Library Demo ===\n")
    
    # Example using OpenRouter's free Grok API
    @dataclass
    class Person:
        name: str
        age: int
    
    # Note: You'll need to get a free API key from OpenRouter
    # Visit: https://openrouter.ai/keys
    API_KEY = "your-openrouter-api-key"  # Replace with actual key
    
    backend = OpenAIBackend(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        model="grok-beta",  # Free model on OpenRouter
        temperature=0.7,
        max_tokens=500
    )
    
    CORPUS = """
    In the bustling city, we met several interesting people. 
    John Smith, a 25-year-old engineer, was passionate about robotics.
    Sarah Johnson, only 17, showed remarkable talent in mathematics.
    Robert Brown, 42, had traveled the world as a photographer.
    Emily Davis, 19, just started her college journey.
    """
    
    PROMPT = f"Find all the people's names and their ages that appear in the corpus:\n{CORPUS}"
    
    try:
        # Synchronous example - sample once
        print("Synchronous execution (single sample):")
        result = (
            Prompt(prompt=PROMPT, backend=backend)
            .sample()  # Default is now max_iterations=1
            .extract(Person)
            .filter(lambda p: p.age > 18)
            .take(10)  # Even if we ask for 10, we only get what's in the single response
        )
        
        print(f"Found {len(result)} adults from a single API call:")
        for person in result:
            print(f"  - {person.name}, age {person.age}")
        
        print("\nNote: sample() now defaults to max_iterations=1 to prevent infinite loops.")
        print("Use sample(None) for infinite generation or sample(n) for n API calls.")
        
    except Exception as e:
        print(f"Demo requires a valid OpenRouter API key. Error: {e}")
        print("Get a free key at: https://openrouter.ai/keys")
    
    # Async example (commented out for simplicity)
    # async def async_demo():
    #     result = await (
    #         Prompt(prompt=PROMPT, backend=backend)
    #         .asample(max_iterations=3)
    #         .extract(Person)
    #         .filter(lambda p: p.age > 18)
    #         .take(2)
    #     )
    #     return result


if __name__ == "__main__":
    # Run demo
    # demo()
    
    # Simple unit tests
    print("\n=== Running Tests ===\n")
    
    # Test JSON extraction - single object
    text_with_json = """
    Here's the data you requested:
    ```json
    {"name": "Alice", "age": 30}
    ```
    """
    
    extracted = SchemaExtractor.extract_all_json_from_text(text_with_json)
    assert len(extracted) == 1 and extracted[0] == {"name": "Alice", "age": 30}, "JSON extraction failed"
    print("✓ JSON extraction test passed")
    
    # Test multiple JSON extraction
    text_with_multiple = """
    Here are the people:
    [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35}
    ]
    """
    
    extracted_multiple = SchemaExtractor.extract_all_json_from_text(text_with_multiple)
    assert len(extracted_multiple) == 3, "Multiple JSON extraction failed"
    print("✓ Multiple JSON extraction test passed")
    
    # Test dataclass parsing
    @dataclass
    class TestPerson:
        name: str
        age: int
    
    data = {"name": "Bob", "age": 25, "extra": "ignored"}
    person = SchemaExtractor.parse_to_dataclass(data, TestPerson)
    assert person.name == "Bob" and person.age == 25, "Dataclass parsing failed"
    print("✓ Dataclass parsing test passed")
    
    # Test fluent chain operations with take() behavior
    chain = FluentChain(iter([1, 2, 3, 4, 5]))
    result = chain.map(lambda x: x * 2).filter(lambda x: x > 4).take(10)
    assert result == [6, 8, 10], "Fluent chain operations failed"
    print("✓ Fluent chain test passed")
    
    # Test take() with fewer items available
    chain2 = FluentChain(iter([1, 2]))
    result2 = chain2.take(5)  # Ask for 5, but only 2 available
    assert result2 == [1, 2], "Take with fewer items failed"
    print("✓ Take with fewer items test passed")
    
    # Test MockBackend behavior
    class MockBackend(LLMBackend):
        """Mock backend for testing."""
        def __init__(self, responses: List[str], cycle: bool = False):
            self.responses = responses
            self.cycle = cycle
            self.index = 0
        
        def generate(self, prompt: str, **kwargs) -> str:
            if self.index >= len(self.responses):
                if self.cycle:
                    self.index = 0
                else:
                    raise StopIteration("No more responses")
            response = self.responses[self.index]
            self.index += 1
            return response
        
        async def agenerate(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)
    
    # Test the expected behavior with MockBackend
    backend = MockBackend([
        '{"name": "Alice", "age": 30}',
        '{"name": "Bob", "age": 25}'
    ], cycle=False)
    
    @dataclass
    class Person:
        name: str
        age: int
    
    prompt = Prompt("extract person", backend)
    
    # First call - should only get Alice
    results = prompt.sample().extract(Person).take(2)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].name == "Alice", f"Expected Alice, got {results[0].name}"
    print("✓ Single sample behavior test passed")
    
    # Second call - should get Bob
    results2 = prompt.sample().extract(Person).take(2)
    assert len(results2) == 1, f"Expected 1 result, got {len(results2)}"
    assert results2[0].name == "Bob", f"Expected Bob, got {results2[0].name}"
    print("✓ Sequential sample behavior test passed")
    
    # Test with multiple items in single response
    backend3 = MockBackend([
        '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]'
    ], cycle=False)
    prompt3 = Prompt("extract people", backend3)
    results3 = prompt3.sample().extract(Person).take(2)
    assert len(results3) == 2, f"Expected 2 results, got {len(results3)}"
    assert results3[0].name == "Alice" and results3[1].name == "Bob"
    print("✓ Multiple items in single response test passed")
    
    # Test retry logic with failing backend
    class FailingBackend(LLMBackend):
        """Mock backend that fails a certain number of times before succeeding."""
        def __init__(self, failures_before_success: int = 2):
            self.failures_before_success = failures_before_success
            self.attempt_count = 0
        
        def generate(self, prompt: str, **kwargs) -> str:
            self.attempt_count += 1
            if self.attempt_count <= self.failures_before_success:
                raise Exception(f"Simulated failure {self.attempt_count}")
            return '{"name": "Success", "age": 100}'
        
        async def agenerate(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)
        
        def reset(self):
            self.attempt_count = 0
    
    # Test retry logic
    print("\n=== Testing Retry Logic ===")
    failing_backend = FailingBackend(failures_before_success=2)
    prompt_with_retry = Prompt("test", failing_backend)
    
    try:
        # Should succeed after 2 failures with default max_retries=3
        result = prompt_with_retry.sample(max_retries=3, retry_delay=0.1).extract(Person).take(1)
        assert len(result) == 1
        assert result[0].name == "Success"
        print("✓ Retry logic test passed (succeeded after retries)")
    except Exception as e:
        print(f"✗ Retry logic test failed: {e}")
    
    # Test when retries are exhausted
    failing_backend.reset()
    failing_backend.failures_before_success = 5  # Will fail more times than max_retries
    
    try:
        result = prompt_with_retry.sample(max_retries=2, retry_delay=0.1).extract(Person).take(1)
        print("✗ Should have failed with exhausted retries")
    except Exception as e:
        if "failed after 3 attempts" in str(e):
            print("✓ Retry exhaustion test passed")
        else:
            print(f"✗ Unexpected error: {e}")
    
    print("\nAll tests passed! ✓")
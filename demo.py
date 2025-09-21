"""A demonstration of the llm-fluent library using OpenRouter."""

"""Fluent-style LLM prompting and response transformation library.

This module provides a fluent interface for working with LLMs, supporting
multiple backends and both synchronous and asynchronous operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import TypeVar

from llm_fluent.backends.base import MockBackend
from llm_fluent.prompt import Prompt, AsyncPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class Person:
    """Example dataclass for extraction."""
    name: str = ""
    age: int = 0


def demo_sync():
    """Demonstrate synchronous usage."""
    print("=== Synchronous Demo ===\n")
    
    # Create mock backend with sample responses
    responses = [
        '{"name": "Alice", "age": 30}',
        '{"name": "Bob", "age": 25}',
        '{"name": "Charlie", "age": 17}',
        '{"name": "David", "age": 40}',
    ]
    backend = MockBackend(responses)
    
    corpus = """
    In our company, we have several employees:
    Alice is 30 years old and works in engineering.
    Bob, age 25, is in marketing.
    Charlie is our 17-year-old intern.
    David, who is 40, manages the team.
    """
    
    prompt = f"Extract people's names and ages from:\n{corpus}\nReturn as JSON."
    
    # Use the fluent interface
    result = (
        Prompt(prompt=prompt, backend=backend)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age > 18)
        .take(2)
    )
    
    print(f"Prompt: {prompt}\n")
    print("Results (adults only, first 2):")
    for person in result:
        print(f"  - {person.name}: {person.age} years old")


async def demo_async():
    """Demonstrate asynchronous usage."""
    print("\n=== Asynchronous Demo ===\n")
    
    # Create mock backend
    responses = [
        '{"name": "Emma", "age": 28}',
        '{"name": "Frank", "age": 35}',
        '{"name": "Grace", "age": 16}',
        '{"name": "Henry", "age": 42}',
    ]
    backend = MockBackend(responses)
    
    prompt = "Generate a random person with name and age as JSON."
    
    # Use the async fluent interface
    result = await (
        AsyncPrompt(prompt=prompt, backend=backend)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age >= 21)
        .take(3)
    )
    
    print(f"Prompt: {prompt}\n")
    print("Results (21+ only, first 3):")
    for person in result:
        print(f"  - {person.name}: {person.age} years old")


def run_tests():
    """Run basic tests."""
    print("\n=== Running Tests ===\n")
    
    # Test 1: Basic extraction
    backend = MockBackend(['{"name": "Test", "age": 25}'], cycle=False)
    result = Prompt("test", backend).sample().extract(Person).take(1)
    assert len(result) == 1
    assert result[0].name == "Test"
    assert result[0].age == 25
    print("✓ Test 1: Basic extraction passed")
    
    # Test 2: Filtering with finite responses
    backend = MockBackend([
        '{"name": "Adult", "age": 30}',
        '{"name": "Minor", "age": 15}',
        '{"name": "Senior", "age": 65}'
    ], cycle=False)
    adults = (
        Prompt("test", backend, max_iterations=3)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age >= 18)
        .take(3)
    )
    assert len(adults) == 2  # Only Adult and Senior pass the filter
    assert all(p.age >= 18 for p in adults)
    print("✓ Test 2: Filtering passed")
    
    # Test 3: Mapping
    backend = MockBackend(['{"name": "John", "age": 25}'], cycle=False)
    names = (
        Prompt("test", backend)
        .sample()
        .extract(Person)
        .map(lambda p: p.name.upper())
        .take(1)
    )
    assert names[0] == "JOHN"
    print("✓ Test 3: Mapping passed")
    
    # Test 4: Chain operations with finite data
    backend = MockBackend([
        '{"name": "Alice", "age": 30}',
        '{"name": "Bob", "age": 20}',
        '{"name": "Charlie", "age": 40}'
    ], cycle=False)
    result = (
        Prompt("test", backend, max_iterations=3)
        .sample()
        .extract(Person)
        .filter(lambda p: p.age > 25)
        .map(lambda p: f"{p.name} ({p.age})")
        .take(5)  # Ask for 5 but only 2 match
    )
    assert len(result) == 2  # Only Alice and Charlie match
    assert "Alice (30)" in result
    assert "Charlie (40)" in result
    print("✓ Test 4: Chain operations passed")
    
    # Test 5: Infinite cycling for production use
    backend = MockBackend(['{"name": "Cycled", "age": 25}'], cycle=True)
    result = (
        Prompt("test", backend)
        .sample()
        .extract(Person)
        .take(3)  # Should get 3 of the same person
    )
    assert len(result) == 3
    assert all(p.name == "Cycled" for p in result)
    print("✓ Test 5: Infinite cycling passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Run synchronous demo
    demo_sync()
    
    # Run asynchronous demo
    asyncio.run(demo_async())
    
    # Run tests
    run_tests()
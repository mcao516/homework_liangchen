"""Unit tests for the Prompt class.

This module contains comprehensive tests for the synchronous Prompt class
and its chain operations.
"""

import unittest
import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import Mock, patch

from llm_fluent.prompt import Prompt, AsyncPrompt
from llm_fluent.backends.base import MockBackend, Backend


@dataclass
class Person:
    """Example dataclass for extraction."""
    name: str = ""
    age: int = 0


@dataclass
class Product:
    """Test dataclass for product extraction."""
    name: str = ""
    price: float = 0.0
    in_stock: bool = True


@dataclass
class ComplexData:
    """Test dataclass with list."""
    id: int = 0
    name: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TestPrompt(unittest.TestCase):
    """Test cases for the Prompt class."""
    
    def test_basic_sample_generation(self):
        """Test 1: Basic sample generation returns Chain with correct data."""
        backend = MockBackend(["Response 1", "Response 2", "Response 3"], cycle=False)
        prompt = Prompt("test prompt", backend, max_iterations=3)
        
        results = prompt.sample().take(3)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response 1")
        self.assertEqual(results[1], "Response 2")
        self.assertEqual(results[2], "Response 3")
    
    def test_extraction_with_valid_json(self):
        """Test 2: Extract structured data from valid JSON responses."""
        backend = MockBackend([
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 25}'
        ], cycle=False)
        prompt = Prompt("extract person", backend)
        
        results = prompt.sample().extract(Person).take(2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].name, "Alice")
        self.assertEqual(results[0].age, 30)
        self.assertEqual(results[1].name, "Bob")
        self.assertEqual(results[1].age, 25)
    
    def test_filter_operation(self):
        """Test 3: Filter operation correctly filters extracted data."""
        backend = MockBackend([
            '{"name": "Adult", "age": 30}',
            '{"name": "Minor", "age": 15}',
            '{"name": "Senior", "age": 65}',
            '{"name": "Teen", "age": 17}'
        ], cycle=False)
        prompt = Prompt("test", backend, max_iterations=4)
        
        adults = prompt.sample().extract(Person).filter(lambda p: p.age >= 18).take(10)
        
        self.assertEqual(len(adults), 2)
        self.assertTrue(all(p.age >= 18 for p in adults))
        self.assertIn("Adult", [p.name for p in adults])
        self.assertIn("Senior", [p.name for p in adults])
    
    def test_map_operation(self):
        """Test 4: Map operation transforms data correctly."""
        backend = MockBackend([
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 25}'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        descriptions = (
            prompt.sample()
            .extract(Person)
            .map(lambda p: f"{p.name} is {p.age} years old")
            .take(2)
        )
        
        self.assertEqual(len(descriptions), 2)
        self.assertEqual(descriptions[0], "Alice is 30 years old")
        self.assertEqual(descriptions[1], "Bob is 25 years old")
    
    def test_chain_operations_combination(self):
        """Test 5: Multiple chain operations work together correctly."""
        backend = MockBackend([
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 20}',
            '{"name": "Charlie", "age": 40}',
            '{"name": "Diana", "age": 35}'
        ], cycle=False)
        prompt = Prompt("test", backend, max_iterations=4)
        
        result = (
            prompt.sample()
            .extract(Person)
            .filter(lambda p: p.age > 25)
            .map(lambda p: p.age)
            .filter(lambda age: age < 40)
            .take(10)
        )
        
        self.assertEqual(sorted(result), [30, 35])
    
    def test_max_iterations_limit(self):
        """Test 6: max_iterations parameter limits the number of samples."""
        backend = MockBackend(["Response"], cycle=True)
        prompt = Prompt("test", backend, max_iterations=5)
        
        results = prompt.sample().take(10)
        
        # Should only get 5 responses despite asking for 10
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r == "Response" for r in results))
    
    def test_extraction_with_malformed_json(self):
        """Test 7: Extraction handles malformed JSON gracefully."""
        backend = MockBackend([
            'Not a JSON',
            'name: John, age: 25',  # Key-value format
            '{"invalid json}',
            '{"name": "Valid", "age": 30}'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        results = prompt.sample().extract(Person).take(4)
        
        self.assertEqual(len(results), 4)
        # First result should be empty/default Person
        self.assertEqual(results[0].name, "")
        self.assertEqual(results[0].age, 0)
        # Second might extract from key-value format
        # Fourth should be valid
        self.assertEqual(results[3].name, "Valid")
        self.assertEqual(results[3].age, 30)
    
    def test_complex_dataclass_extraction(self):
        """Test 8: Extract complex nested dataclass structures."""
        backend = MockBackend([
            '{"id": 1, "name": "language", "tags": ["python", "testing"]}',
            '{"id": 2, "name": "null & void", "tags": []}'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        results = prompt.sample().extract(ComplexData).take(2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, 1)
        self.assertEqual(results[0].name, "language")
        self.assertEqual(results[0].tags, ["python", "testing"])
        self.assertEqual(results[1].id, 2)
        self.assertEqual(results[1].name, "null & void")
        self.assertEqual(results[1].tags, [])
    
    def test_retry_on_failure(self):
        """Test 9: Retry mechanism works on backend failures."""
        # Create a backend that fails twice then succeeds
        backend = Mock(spec=Backend)
        backend.generate = Mock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "Success response"
        ])
        
        prompt = Prompt("test", backend, max_retries=3, retry_delay=0.01)
        
        with patch('time.sleep'):
            results = prompt.sample().take(1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Success response")
        self.assertEqual(backend.generate.call_count, 3)
    
    def test_finite_data_exhaustion(self):
        """Test 10: Properly handles exhaustion of finite data source."""
        backend = MockBackend([
            '{"name": "Only", "age": 25}'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        # Try to take more than available
        results = prompt.sample().extract(Person).take(5)
        
        # Should only get 1 result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Only")
        self.assertEqual(results[0].age, 25)


class TestAsyncPrompt(unittest.TestCase):
    """Test cases for the AsyncPrompt class."""
    
    def test_async_basic_sample(self):
        """Test 11: Async sample generation works correctly."""
        async def run_test():
            backend = MockBackend(["Async 1", "Async 2"], cycle=False)
            prompt = AsyncPrompt("test", backend)
            
            results = await prompt.sample().take(2)
            
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0], "Async 1")
            self.assertEqual(results[1], "Async 2")
        
        asyncio.run(run_test())
    
    def test_async_chain_operations(self):
        """Test 12: Async chain operations work correctly."""
        async def run_test():
            backend = MockBackend([
                '{"name": "Alice", "age": 30}',
                '{"name": "Bob", "age": 20}',
                '{"name": "Charlie", "age": 40}'
            ], cycle=False)
            prompt = AsyncPrompt("test", backend, max_iterations=3)
            
            results = await (
                prompt.sample()
                .extract(Person)
                .filter(lambda p: p.age > 25)
                .map(lambda p: p.name.upper())
                .take(5)
            )
            
            self.assertEqual(len(results), 2)
            self.assertIn("ALICE", results)
            self.assertIn("CHARLIE", results)
        
        asyncio.run(run_test())
    
    # def test_async_collect_all(self):
    #     """Test 13: Async collect method gathers all results."""
    #     async def run_test():
    #         backend = MockBackend(["A", "B", "C"], cycle=False)
    #         prompt = AsyncPrompt("test", backend)
            
    #         results = await prompt.sample().collect()
            
    #         self.assertEqual(results, ["A", "B", "C"])
        
    #     asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
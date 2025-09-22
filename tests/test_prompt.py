"""Unit tests for the Prompt class.

This module contains tests for the synchronous Prompt class
and its chain operations.
"""

import unittest
import asyncio
from dataclasses import dataclass
from typing import List
from unittest.mock import Mock, patch

from llm_fluent import Prompt
from llm_fluent.backends.base import MockBackend, LLMBackend

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None


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
        prompt = Prompt("test prompt", backend)
        
        results = prompt.sample().take(3)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Response 1")

    def test_extraction_with_valid_json(self):
        """Test 2: Extract structured data from valid JSON responses."""
        backend = MockBackend([
            '{"name": "Alice", "age": 30}',
            '{"name": "Bob", "age": 25}'
        ], cycle=False)
        prompt = Prompt("extract person", backend)
        
        results = prompt.sample().extract(Person).take(2)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Alice")
        self.assertEqual(results[0].age, 30)
    
    def test_filter_operation(self):
        """Test 3: Filter operation correctly filters extracted data."""
        backend = MockBackend([
            '[{"name": "Adult", "age": 30}, {"name": "Minor", "age": 15}, {"name": "Senior", "age": 65}]',
            '[{"name": "Jackie", "age": 32}]'
        ], cycle=False)
        prompt = Prompt("Extract all people from the text", backend)
        
        adults = prompt.sample().extract(Person).filter(lambda p: p.age >= 18).take(10)
        
        self.assertEqual(len(adults), 2)
        self.assertTrue(all(p.age >= 18 for p in adults))
        self.assertIn("Adult", [p.name for p in adults])
        self.assertIn("Senior", [p.name for p in adults])
    
    def test_map_operation(self):
        """Test 4: Map operation transforms data correctly."""
        backend = MockBackend([
            '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]',
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        descriptions = (
            prompt.sample()
            .extract(Person)
            .map(lambda p: f"{p.name} is {p.age} years old")
            .collect()
        )
        
        self.assertEqual(len(descriptions), 2)
        self.assertEqual(descriptions[0], "Alice is 30 years old")
        self.assertEqual(descriptions[1], "Bob is 25 years old")
    
    def test_chain_operations_combination(self):
        """Test 5: Multiple chain operations work together correctly."""
        backend = MockBackend([
            '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 20}, \
                {"name": "Charlie", "age": 40}, {"name": "Diana", "age": 35}]'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
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
        prompt = Prompt("test", backend)
        
        results = prompt.sample().take(10)
        
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0] == "Response")
    
    def test_extraction_with_malformed_json(self):
        """Test 7: Extraction handles malformed JSON gracefully."""
        backend = MockBackend(['waht the big deal?'], cycle=True)
        prompt = Prompt("test", backend)
        
        results = prompt.sample().extract(Person).take(2)
        
        self.assertEqual(len(results), 0)
    
    def test_complex_dataclass_extraction(self):
        """Test 8: Extract complex nested dataclass structures."""
        backend = MockBackend([
            '[{"id": 1, "name": "language", "tags": ["python", "testing"]}, \
                {"id": 2, "name": "null & void", "tags": []}]'
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
        # Create a LLM backend that fails twice then succeeds
        backend = Mock(spec=LLMBackend)
        backend.generate = Mock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            "Success response"
        ])
        
        prompt = Prompt("test", backend)
        
        with patch('time.sleep'):
            results = prompt.sample(max_retries=3, retry_delay=0.01).take(1)
        
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

    def test_extraction_with_json_schema(self):
        """Test 14: Properly extract given a json scheme."""
        json_schema = {"item_name": "string", "item_id": "integer"}
        backend = MockBackend([
            '[{"item_name": "Test Item", "item_id": 987, "is_active": true}]'
        ], cycle=False)
        prompt = Prompt("test", backend)
        
        # Try to take more than available
        results = prompt.sample().extract(json_schema).collect()
        
        # Should only get 1 result
        self.assertEqual(len(results), 1)
        self.assertTrue(isinstance(results[0], dict))
        self.assertEqual(results[0]["item_name"], "Test Item")
        self.assertEqual(results[0]["item_id"], 987)
        self.assertTrue(results[0]["is_active"])

    def test_extraction_with_pydantic_schema(self):
        """Test 15: Properly extract given a pydantic scheme."""
        if HAS_PYDANTIC:
            class Product(BaseModel):
                name: str
                price: float
                in_stock: bool

            mock_response_pydantic = '{"name": "Super Widget", "price": 19.99, "in_stock": true, "extra_field": "should be ignored"}'
            
            backend_for_pydantic = MockBackend([mock_response_pydantic])
            prompt_for_pydantic = Prompt("extract product", backend_for_pydantic)

            result_pydantic = prompt_for_pydantic.sample().extract(Product).collect()

            self.assertEqual(len(result_pydantic), 1)
            extracted_product = result_pydantic[0]
            self.assertTrue(isinstance(extracted_product, Product))
            self.assertEqual(extracted_product.name, "Super Widget")
            self.assertEqual(extracted_product.price, 19.99)
            self.assertTrue(not hasattr(extracted_product, "extra_field"))
        else:
            print("! Pydantic not installed, skipping Pydantic schema extraction test")


class TestAsyncPrompt(unittest.TestCase):
    """Test cases for the AsyncPrompt class."""
    
    def test_async_basic_sample(self):
        """Test 11: Async sample generation works correctly."""
        async def run_test():
            backend = MockBackend(["Async 1", "Async 2"], cycle=True)
            prompt = Prompt("test", backend)
            
            results = await prompt.asample().take(2)
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], "Async 1")
        
        asyncio.run(run_test())
    
    def test_async_chain_operations(self):
        """Test 12: Async chain operations work correctly."""
        async def run_test():
            backend = MockBackend([
                '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 20}, {"name": "Charlie", "age": 40}]',
            ], cycle=False)
            prompt = Prompt("test", backend)

            results = await (
                prompt.asample()
                .extract(Person)
                .filter(lambda p: p.age > 25)
                .map(lambda p: p.name.upper())
                .take(5)
            )
            
            self.assertEqual(len(results), 2)
            self.assertIn("ALICE", results)
            self.assertIn("CHARLIE", results)
        
        asyncio.run(run_test())
    
    def test_async_collect_all(self):
        """Test 13: Async collect method gathers all results."""
        async def run_test():
            backend = MockBackend(["A", "B", "C"], cycle=True)
            prompt = Prompt("test", backend)
            
            results = await prompt.asample().collect()
            
            self.assertEqual(results, ["A"])
        
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
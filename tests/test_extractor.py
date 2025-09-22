"""Unit tests for SchemaExtractor functionality.
"""

import unittest
from dataclasses import dataclass
from typing import Optional

from llm_fluent import SchemaExtractor


class TestSchemaExtractor(unittest.TestCase):
    """Test suite for SchemaExtractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        @dataclass
        class TestPerson:
            name: str
            age: int
            city: Optional[str] = None
        
        self.TestPerson = TestPerson
    
    def test_extract_json_from_markdown_code_block(self):
        """Test extraction of JSON from markdown code blocks."""
        # Test with json language tag
        text_with_json_tag = """
        Here's the result:
        ```json
        {"name": "Alice", "age": 30, "city": "NYC"}
        ```
        That's the data.
        """
        result = SchemaExtractor.extract_all_json_from_text(text_with_json_tag)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {"name": "Alice", "age": 30, "city": "NYC"})
        
        # Test without language tag
        text_without_tag = """
        Result:
        ```
        {"name": "Bob", "age": 25}
        ```
        """
        result = SchemaExtractor.extract_all_json_from_text(text_without_tag)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {"name": "Bob", "age": 25})
    
    def test_extract_multiple_json_objects(self):
        """Test extraction of multiple JSON objects from text."""
        # Test with array of objects
        text_with_array = """
        Found these people:
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
        """
        result = SchemaExtractor.extract_all_json_from_text(text_with_array)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[1]["name"], "Bob")
        self.assertEqual(result[2]["name"], "Charlie")
        
        # Test with multiple separate JSON objects
        text_with_multiple = """
        First person: {"name": "Alice", "age": 30}
        Second person: {"name": "Bob", "age": 25}
        Some text in between that should be ignored.
        Third: {"name": "Charlie", "age": 35}
        """
        result = SchemaExtractor.extract_all_json_from_text(text_with_multiple)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[1]["name"], "Bob")
        self.assertEqual(result[2]["name"], "Charlie")
    
    def test_extract_nested_and_complex_json(self):
        """Test extraction of nested and complex JSON structures."""
        # Nested objects
        text_with_nested = """
        Complex data:
        {
            "person": {
                "name": "Alice",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "NYC"
                }
            },
            "skills": ["Python", "JavaScript", "SQL"]
        }
        """
        result = SchemaExtractor.extract_all_json_from_text(text_with_nested)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["person"]["name"], "Alice")
        self.assertEqual(result[0]["person"]["address"]["city"], "NYC")
        self.assertEqual(len(result[0]["skills"]), 3)
        
        # Mixed content with noise
        text_with_noise = """
        Some random text here { this is not json }
        But this is: {"valid": true, "number": 42}
        And some more noise: {invalid json without quotes}
        Another valid one: {"name": "Test", "values": [1, 2, 3]}
        """
        result = SchemaExtractor.extract_all_json_from_text(text_with_noise)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["valid"], True)
        self.assertEqual(result[1]["name"], "Test")
    
    def test_create_extraction_prompt(self):
        """Test extraction prompt generation for different schema types."""
        original_prompt = "Find all people"
        
        # Test with dataclass
        prompt_with_dataclass = SchemaExtractor.create_extraction_prompt(
            original_prompt, self.TestPerson
        )
        self.assertIn("Find all people", prompt_with_dataclass)
        self.assertIn("JSON", prompt_with_dataclass)
        self.assertIn("name", prompt_with_dataclass)
        self.assertIn("age", prompt_with_dataclass)
        
        # Test with dict schema
        dict_schema = {"name": "string", "age": "number", "active": "boolean"}
        prompt_with_dict = SchemaExtractor.create_extraction_prompt(
            original_prompt, dict_schema
        )
        self.assertIn("Find all people", prompt_with_dict)
        self.assertIn('"name": "string"', prompt_with_dict)
        self.assertIn('"age": "number"', prompt_with_dict)
        self.assertIn('"active": "boolean"', prompt_with_dict)
        
        # Test with pydantic if available
        try:
            from pydantic import BaseModel
            
            class PydanticPerson(BaseModel):
                name: str
                age: int
                email: Optional[str] = None
            
            prompt_with_pydantic = SchemaExtractor.create_extraction_prompt(
                original_prompt, PydanticPerson
            )
            self.assertIn("Find all people", prompt_with_pydantic)
            self.assertIn("JSON", prompt_with_pydantic)
            # Pydantic generates a full JSON schema
            self.assertIn("properties", prompt_with_pydantic.lower() or prompt_with_pydantic)
        except ImportError:
            # Skip pydantic test if not installed
            pass


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaExtractor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
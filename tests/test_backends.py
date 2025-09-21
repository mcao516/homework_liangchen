"""Unit tests for Backend classes.

This module contains comprehensive tests for all Backend implementations
including MockBackend, OpenAIBackend, and AnthropicBackend.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from llm_fluent.backends.base import Backend
from llm_fluent import MockBackend, OpenAIBackend, AnthropicBackend


class TestMockBackend(unittest.TestCase):
    """Test cases for the MockBackend class."""
    
    def test_single_response_generation(self):
        """Test 1: MockBackend generates single response correctly."""
        backend = MockBackend(["Test response"])
        
        response = backend.generate("any prompt")
        
        self.assertEqual(response, "Test response")
    
    def test_multiple_responses_with_cycling(self):
        """Test 2: MockBackend cycles through multiple responses."""
        backend = MockBackend(["First", "Second", "Third"], cycle=True)
        
        responses = [backend.generate("prompt") for _ in range(5)]
        
        self.assertEqual(responses, ["First", "Second", "Third", "First", "Second"])
    
    def test_finite_responses_without_cycling(self):
        """Test 3: MockBackend stops after exhausting responses when cycle=False."""
        backend = MockBackend(["One", "Two"], cycle=False)
        
        response1 = backend.generate("prompt")
        response2 = backend.generate("prompt")
        
        self.assertEqual(response1, "One")
        self.assertEqual(response2, "Two")
        
        with self.assertRaises(StopIteration):
            backend.generate("prompt")
    
    def test_async_generation(self):
        """Test 4: MockBackend async generation works correctly."""
        async def run_test():
            backend = MockBackend(["Async response"])
            
            response = await backend.agenerate("prompt")
            
            self.assertEqual(response, "Async response")
        
        asyncio.run(run_test())
    
    def test_default_response(self):
        """Test 5: MockBackend uses default response when none provided."""
        backend = MockBackend()
        
        response = backend.generate("prompt")
        
        self.assertEqual(response, "Mock response")
    
    def test_call_count_tracking(self):
        """Test 6: MockBackend tracks call count correctly."""
        backend = MockBackend(["A", "B"], cycle=True)
        
        self.assertEqual(backend.call_count, 0)
        backend.generate("prompt")
        self.assertEqual(backend.call_count, 1)
        backend.generate("prompt")
        self.assertEqual(backend.call_count, 2)
        backend.generate("prompt")
        self.assertEqual(backend.call_count, 3)


class TestOpenAIBackend(unittest.TestCase):
    """Test cases for the OpenAIBackend class."""
    
    def test_openai_initialization_with_base_url(self, mock_openai):
        """Test 7: OpenAIBackend initializes correctly with custom base URL."""
        mock_client = Mock()
        mock_async_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.AsyncOpenAI.return_value = mock_async_client
        
        backend = OpenAIBackend(
            api_key="test-key",
            model="gpt-4",
            base_url="https://custom.api.com",
            temperature=0.5,
            max_tokens=500
        )
        
        self.assertEqual(backend.model, "gpt-4")
        self.assertEqual(backend.temperature, 0.5)
        self.assertEqual(backend.max_tokens, 500)
        mock_openai.OpenAI.assert_called_once_with(
            api_key="test-key", 
            base_url="https://custom.api.com"
        )
    
    def test_openai_generate_success(self, mock_openai):
        """Test 8: OpenAIBackend generates response successfully."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="AI response"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.AsyncOpenAI.return_value = Mock()
        
        backend = OpenAIBackend(api_key="test-key")
        response = backend.generate("Test prompt")
        
        self.assertEqual(response, "AI response")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=1000
        )
    
    def test_openai_generate_with_custom_params(self, mock_openai):
        """Test 9: OpenAIBackend accepts custom parameters in generate."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Custom response"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.AsyncOpenAI.return_value = Mock()
        
        backend = OpenAIBackend(api_key="test-key")
        response = backend.generate(
            "Test prompt", 
            temperature=0.2, 
            max_tokens=200
        )
        
        self.assertEqual(response, "Custom response")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.2,
            max_tokens=200
        )
    
    def test_openai_generate_error_handling(self, mock_openai):
        """Test 10: OpenAIBackend handles errors correctly."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.AsyncOpenAI.return_value = Mock()
        
        backend = OpenAIBackend(api_key="test-key")
        
        with self.assertRaises(Exception) as context:
            backend.generate("Test prompt")
        
        self.assertIn("API Error", str(context.exception))
    
    def test_openai_import_error(self):
        """Test 11: OpenAIBackend raises ImportError when openai not installed."""
        with self.assertRaises(ImportError) as context:
            OpenAIBackend(api_key="test-key")
        
        self.assertIn("openai package is not installed", str(context.exception))
    
    def test_openai_async_generate(self, mock_openai):
        """Test 12: OpenAIBackend async generation works correctly."""
        async def run_test():
            # Setup async mock response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Async AI response"))]
            
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = Mock()
            mock_openai.AsyncOpenAI.return_value = mock_async_client
            
            backend = OpenAIBackend(api_key="test-key")
            response = await backend.agenerate("Test prompt")
            
            self.assertEqual(response, "Async AI response")
        
        asyncio.run(run_test())


class TestAnthropicBackend(unittest.TestCase):
    """Test cases for the AnthropicBackend class."""
    
    def test_anthropic_initialization(self, mock_anthropic):
        """Test 13: AnthropicBackend initializes correctly."""
        mock_client = Mock()
        mock_async_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client
        
        backend = AnthropicBackend(
            api_key="test-key",
            model="claude-3-opus-20240229",
            max_tokens=2000
        )
        
        self.assertEqual(backend.model, "claude-3-opus-20240229")
        self.assertEqual(backend.max_tokens, 2000)
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")
    
    def test_anthropic_generate_success(self, mock_anthropic):
        """Test 14: AnthropicBackend generates response successfully."""
        # Setup mock response
        mock_content = Mock(text="Claude response")
        mock_response = Mock(content=[mock_content])
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = Mock()
        
        backend = AnthropicBackend(api_key="test-key")
        response = backend.generate("Test prompt")
        
        self.assertEqual(response, "Claude response")
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Test prompt"}]
        )
    
    def test_anthropic_error_handling(self, mock_anthropic):
        """Test 15: AnthropicBackend handles errors correctly."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Anthropic API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        mock_anthropic.AsyncAnthropic.return_value = Mock()
        
        backend = AnthropicBackend(api_key="test-key")
        
        with self.assertRaises(Exception) as context:
            backend.generate("Test prompt")
        
        self.assertIn("Anthropic API Error", str(context.exception))
    
    def test_anthropic_import_error(self):
        """Test 16: AnthropicBackend raises ImportError when anthropic not installed."""
        with self.assertRaises(ImportError) as context:
            AnthropicBackend(api_key="test-key")
        
        self.assertIn("anthropic package is not installed", str(context.exception))


class TestBackendInterface(unittest.TestCase):
    """Test cases for the Backend abstract base class."""
    
    def test_backend_is_abstract(self):
        """Test 17: Backend class cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            Backend()
    
    def test_backend_requires_implementation(self):
        """Test 18: Backend subclass must implement required methods."""
        class IncompleteBackend(Backend):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteBackend()
    
    def test_backend_complete_implementation(self):
        """Test 19: Backend subclass with all methods can be instantiated."""
        class CompleteBackend(Backend):
            def generate(self, prompt: str, **kwargs) -> str:
                return "test"
            
            async def agenerate(self, prompt: str, **kwargs) -> str:
                return "test"
        
        backend = CompleteBackend()
        self.assertEqual(backend.generate("test"), "test")
    
    def test_backend_inheritance(self):
        """Test 20: All backend implementations inherit from Backend ABC."""
        self.assertTrue(issubclass(MockBackend, Backend))
        self.assertTrue(issubclass(OpenAIBackend, Backend))
        self.assertTrue(issubclass(AnthropicBackend, Backend))


if __name__ == "__main__":
    unittest.main(verbosity=2)
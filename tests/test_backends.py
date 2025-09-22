"""Unit tests for backend implementations.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from llm_fluent import OpenAIBackend, AnthropicBackend


class TestBackends(unittest.IsolatedAsyncioTestCase):
    """Test suite for LLM backend implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prompt = "Test prompt"
        self.test_response = "Test response"
    
    def test_openai_backend_initialization(self):
        """Test OpenAI backend initialization with various parameters."""

        with patch('openai.OpenAI') as mock_openai:
            backend = OpenAIBackend(api_key="test-key")
            self.assertEqual(backend.model, "gpt-4o-mini")
            self.assertEqual(backend.temperature, 0.7)
            self.assertEqual(backend.max_tokens, 1000)
            mock_openai.assert_called_once_with(api_key="test-key")
        
        # Test with custom parameters
        with patch('openai.OpenAI') as mock_openai:
            backend = OpenAIBackend(
                api_key="test-key",
                base_url="https://custom.api.com",
                model="gpt-4",
                temperature=0.5,
                max_tokens=2000
            )
            self.assertEqual(backend.model, "gpt-4")
            self.assertEqual(backend.temperature, 0.5)
            self.assertEqual(backend.max_tokens, 2000)
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.com"
            )
    
    def test_openai_backend_generate(self):
        """Test OpenAI backend synchronous generation."""
        with patch('openai.OpenAI') as mock_openai_class:
            # Set up mock client
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Set up mock response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=self.test_response))]
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create backend and generate
            backend = OpenAIBackend(api_key="test-key", model="gpt-4o-mini")
            result = backend.generate(self.test_prompt)
            
            # Verify the result
            self.assertEqual(result, self.test_response)
            
            # Verify API was called correctly
            mock_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": self.test_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
    
    def test_anthropic_backend_initialization_and_generation(self):
        """Test Anthropic backend initialization and generation."""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Set up mock client
            mock_client = Mock()
            mock_anthropic_class.return_value = mock_client
            
            # Set up mock response
            mock_response = Mock()
            mock_response.content = [Mock(text=self.test_response)]
            mock_client.messages.create.return_value = mock_response
            
            # Create backend and test initialization
            backend = AnthropicBackend(
                api_key="test-key",
                model="claude-3-opus",
                max_tokens=1500,
                temperature=0.8
            )
            
            self.assertEqual(backend.model, "claude-3-opus")
            self.assertEqual(backend.max_tokens, 1500)
            self.assertEqual(backend.temperature, 0.8)
            
            # Test generation
            result = backend.generate(self.test_prompt)
            self.assertEqual(result, self.test_response)
            
            # Verify API was called correctly
            mock_client.messages.create.assert_called_once_with(
                model="claude-3-opus",
                max_tokens=1500,
                temperature=0.8,
                messages=[{"role": "user", "content": self.test_prompt}]
            )
    
    @patch('asyncio.sleep', return_value=None)
    async def test_async_generation_both_backends(self, mock_sleep):
        """Test asynchronous generation for both OpenAI and Anthropic backends."""
        # Test OpenAI async
        with patch('openai.AsyncOpenAI') as mock_async_openai:
            mock_client = AsyncMock()
            mock_async_openai.return_value = mock_client
            
            # Mock async response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="async response"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            backend = OpenAIBackend(api_key="test-key")
            result = await backend.agenerate(self.test_prompt)
            self.assertEqual(result, "async response")
        
        # Test Anthropic async
        with patch('anthropic.AsyncAnthropic') as mock_async_anthropic:
            mock_client = AsyncMock()
            mock_async_anthropic.return_value = mock_client
            
            # Mock async response
            mock_response = Mock()
            mock_response.content = [Mock(text="async anthropic response")]
            mock_client.messages.create.return_value = mock_response
            
            backend = AnthropicBackend(api_key="test-key")
            result = await backend.agenerate(self.test_prompt)
            self.assertEqual(result, "async anthropic response")


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestBackends))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    if sys.version_info >= (3, 7):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    success = run_tests()
    sys.exit(0 if success else 1)
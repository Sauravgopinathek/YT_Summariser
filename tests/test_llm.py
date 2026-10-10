"""Unit tests for modules.llm using mocked OpenAI client."""

import unittest
from unittest.mock import MagicMock, patch

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from modules.llm import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    NVIDIAClient,
    verify_nvidia_connection,
)


class TestNVIDIAClient(unittest.TestCase):
    """Test suite for NVIDIA NIM LLM client interface and error mapping."""

    def test_missing_api_key_raises_configuration_error(self):
        """Test that missing API key raises LLMConfigurationError."""
        with patch("modules.llm.get_nvidia_api_key", return_value=""):
            client = NVIDIAClient(api_key="")
            client.api_key = ""
            with self.assertRaises(LLMConfigurationError):
                client.generate(system_prompt="sys", user_prompt="usr")


    @patch("modules.llm.OpenAI")
    def test_successful_generation(self, mock_openai_cls):
        """Test successful completion response from NIM endpoint."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Summary of video content."
        mock_response.choices = [mock_choice]

        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-dummy-key-for-test-1234")
        output = client.generate(
            system_prompt="You are a helpful summarizer.",
            user_prompt="Summarize this transcript.",
            temperature=0.3,
            max_tokens=500,
        )

        self.assertEqual(output, "Summary of video content.")
        mock_instance.chat.completions.create.assert_called_once()
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.3)
        self.assertEqual(call_kwargs["max_tokens"], 500)
        self.assertEqual(len(call_kwargs["messages"]), 2)

    @patch("modules.llm.OpenAI")
    def test_test_connection_handshake(self, mock_openai_cls):
        """Test test_connection sends the handshake prompt and succeeds."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "CONNECTION_SUCCESS"
        mock_response.choices = [mock_choice]

        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-dummy-key-1234")
        response = client.test_connection()
        self.assertEqual(response, "CONNECTION_SUCCESS")
        self.assertTrue(verify_nvidia_connection(client))

    @patch("modules.llm.OpenAI")
    def test_authentication_error_mapping(self, mock_openai_cls):
        """Test 401 AuthenticationError maps to LLMAuthenticationError."""
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-invalid-key")
        with self.assertRaises(LLMAuthenticationError):
            client.generate(system_prompt="sys", user_prompt="usr")

    @patch("modules.llm.OpenAI")
    def test_rate_limit_error_mapping(self, mock_openai_cls):
        """Test 429 RateLimitError maps to LLMRateLimitError."""
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit reached",
            response=MagicMock(status_code=429),
            body=None,
        )
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-dummy-key")
        with self.assertRaises(LLMRateLimitError):
            client.generate(system_prompt="sys", user_prompt="usr")

    @patch("modules.llm.OpenAI")
    def test_timeout_error_mapping(self, mock_openai_cls):
        """Test APITimeoutError maps to LLMTimeoutError."""
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-dummy-key")
        with self.assertRaises(LLMTimeoutError):
            client.generate(system_prompt="sys", user_prompt="usr")

    @patch("modules.llm.OpenAI")
    def test_connection_error_mapping(self, mock_openai_cls):
        """Test APIConnectionError maps to LLMConnectionError."""
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        mock_openai_cls.return_value = mock_instance

        client = NVIDIAClient(api_key="nvapi-dummy-key")
        with self.assertRaises(LLMConnectionError):
            client.generate(system_prompt="sys", user_prompt="usr")


if __name__ == "__main__":
    unittest.main()

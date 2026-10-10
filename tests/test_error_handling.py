"""Unit tests for centralized error handling and secret sanitization."""

import unittest

from modules.exceptions import (
    AppError,
    CaptionsDisabledError,
    ChunkingError,
    EmptyLLMResponseError,
    EmptyTranscriptError,
    EnglishTranscriptNotFoundError,
    InvalidVideoIDError,
    InvalidYouTubeURLError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMContextOverflowError,
    LLMRateLimitError,
    LLMTimeoutError,
    MarkdownValidationError,
    MissingRequiredSectionError,
    TranscriptAPIError,
    TranscriptTooLongError,
    VideoUnavailableError,
    sanitize_secrets,
)
from utils.error_handler import format_user_error


class TestCentralizedErrorHandling(unittest.TestCase):
    """Test suite verifying error translation and secret protection across all 16 failure modes."""

    def test_sanitize_secrets(self):
        """Test that API keys and bearer tokens are properly masked."""
        secret_text = "Failed with key nvapi-abc1234567890XYZ and Bearer token-sec-998877"
        sanitized = sanitize_secrets(secret_text)
        self.assertNotIn("nvapi-abc1234567890XYZ", sanitized)
        self.assertIn("nvapi-***", sanitized)
        self.assertNotIn("token-sec-998877", sanitized)

    def test_all_16_error_mappings(self):
        """Test human-readable title and message for all 16 required failure conditions."""
        cases = [
            (InvalidYouTubeURLError("Bad URL"), "Invalid YouTube URL"),
            (InvalidVideoIDError("Short ID"), "Invalid Video ID"),
            (CaptionsDisabledError("Disabled"), "Subtitles Disabled"),
            (EnglishTranscriptNotFoundError("No EN"), "English Captions Unavailable"),
            (VideoUnavailableError("Private"), "Video Unavailable"),
            (TranscriptAPIError("Network drop"), "Transcript Service Error"),
            (EmptyTranscriptError("No speech"), "Empty Transcript"),
            (TranscriptTooLongError("100k words"), "Transcript Exceeds Limit"),
            (ChunkingError("Overlap too big"), "Context Chunking Error"),
            (LLMConfigurationError("Missing key"), "NVIDIA Configuration Missing"),
            (LLMAuthenticationError("401 key rejected"), "NVIDIA Authentication Failure (401)"),
            (LLMRateLimitError("429 cap hit"), "NVIDIA Rate Limit / Quota Exceeded (429)"),
            (LLMTimeoutError("Timed out"), "LLM Request Timeout"),
            (LLMConnectionError("Cannot connect"), "LLM Connection Failure"),
            (LLMContextOverflowError("Prompt too long"), "Model Context Overflow"),
            (EmptyLLMResponseError("No choices"), "Empty LLM Response"),
            (MissingRequiredSectionError("No Action Items"), "Incomplete Notes Generated"),
            (MarkdownValidationError("Malformed"), "Malformed Markdown Output"),
        ]

        for err_instance, expected_title in cases:
            with self.subTest(err=type(err_instance).__name__):
                title, msg = format_user_error(err_instance)
                self.assertEqual(title, expected_title)
                self.assertIsInstance(msg, str)
                self.assertGreater(len(msg), 5)

    def test_secret_never_exposed_in_error_message(self):
        """Test that even if raw exception contains an API key, it is redacted."""
        raw_err = AppError("Raw server failure with key nvapi-secret1234567890")
        title, msg = format_user_error(raw_err)
        self.assertNotIn("nvapi-secret1234567890", msg)
        self.assertIn("nvapi-***", msg)


if __name__ == "__main__":
    unittest.main()

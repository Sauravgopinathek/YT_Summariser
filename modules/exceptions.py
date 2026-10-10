"""Centralized exception hierarchy and error sanitization for the application.

Ensures structured error handling, domain-specific exception propagation,
and protection against leaking secrets in error strings and logs.
"""

import re
from typing import Optional


def sanitize_secrets(text: str) -> str:
    """Mask sensitive tokens and keys in error messages and logs.

    Args:
        text: Input string potentially containing secret tokens.

    Returns:
        Sanitized string with keys replaced by asterisks.
    """
    if not text:
        return ""
    # Mask NVIDIA NIM keys: nvapi-...
    cleaned = re.sub(r"nvapi-[A-Za-z0-9_-]{8,}", "nvapi-***", str(text))
    # Mask standard API keys / Bearer tokens
    cleaned = re.sub(r"(Bearer\s+)[A-Za-z0-9_\.\-]{8,}", r"\1***", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(key=)[A-Za-z0-9_\.\-]{8,}", r"\1***", cleaned, flags=re.IGNORECASE)
    return cleaned


class AppError(Exception):
    """Base application exception for all domain-specific errors."""

    def __init__(self, message: str, user_friendly_message: Optional[str] = None):
        sanitized_msg = sanitize_secrets(message)
        super().__init__(sanitized_msg)
        self.user_friendly_message = (
            sanitize_secrets(user_friendly_message) if user_friendly_message else sanitized_msg
        )


# ---------------------------------------------------------------------------
# URL & Video ID Exceptions
# ---------------------------------------------------------------------------

class InvalidYouTubeURLError(AppError, ValueError):
    """Raised when URL is malformed or not a recognized YouTube domain."""
    pass


class InvalidVideoIDError(AppError, ValueError):
    """Raised when YouTube video ID is not exactly 11 valid characters."""
    pass


# ---------------------------------------------------------------------------
# Transcript Exceptions
# ---------------------------------------------------------------------------

class TranscriptError(AppError):
    """Base exception for transcript acquisition errors."""
    pass


class CaptionsDisabledError(TranscriptError):
    """Raised when subtitles/closed captions are disabled on the video."""
    pass


class VideoUnavailableError(TranscriptError):
    """Raised when the video cannot be found, is private, or has been removed."""
    pass


class EnglishTranscriptNotFoundError(TranscriptError):
    """Raised when no English subtitle track (manual or auto-generated) is found."""
    pass


class TranscriptAPIError(TranscriptError):
    """Raised when the YouTube Transcript API encounters a network or service failure."""
    pass


class EmptyTranscriptError(TranscriptError, ValueError):
    """Raised when a transcript contains no text or is empty after cleaning."""
    pass


class TranscriptTooLongError(TranscriptError, ValueError):
    """Raised when transcript exceeds safe processing limits."""
    pass


# ---------------------------------------------------------------------------
# Chunking Exceptions
# ---------------------------------------------------------------------------

class ChunkingError(AppError, ValueError):
    """Raised when transcript chunking fails due to invalid parameters or partition failure."""
    pass


# ---------------------------------------------------------------------------
# LLM & NIM Exceptions
# ---------------------------------------------------------------------------

class LLMError(AppError):
    """Base exception for LLM inference failures."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when API credentials or required settings are missing."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (HTTP 401: Invalid API Key)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when API rate limits or quota caps are hit (HTTP 429)."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an API call times out."""
    pass


class LLMConnectionError(LLMError):
    """Raised when network connectivity fails."""
    pass


class LLMAPIError(LLMError):
    """Raised when the LLM service returns an unhandled error status."""
    pass


class LLMContextOverflowError(LLMError):
    """Raised when prompt tokens exceed model context window."""
    pass


class EmptyLLMResponseError(LLMError):
    """Raised when the model returns an empty reply or choices array."""
    pass


# ---------------------------------------------------------------------------
# Notes Formatting & Validation Exceptions
# ---------------------------------------------------------------------------

class MarkdownValidationError(AppError, ValueError):
    """Raised when generated notes fail required Markdown structural verification."""
    pass


class MissingRequiredSectionError(MarkdownValidationError):
    """Raised when Executive Summary, Key Concepts, or Action Items is absent."""
    pass

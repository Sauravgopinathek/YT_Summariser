"""Centralized error handling and user-facing message translation utility.

Converts domain exceptions into structured, human-readable UI notifications
while ensuring internal tokens, keys, and tracebacks remain completely hidden.
"""

import logging
from typing import Tuple

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
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    MarkdownValidationError,
    MissingRequiredSectionError,
    TranscriptAPIError,
    TranscriptError,
    TranscriptTooLongError,
    VideoUnavailableError,
    sanitize_secrets,
)

logger = logging.getLogger(__name__)


def format_user_error(err: Exception) -> Tuple[str, str]:
    """Translate an exception into a human-friendly title and actionable message.

    Args:
        err: Caught exception instance.

    Returns:
        Tuple of (Title, Description).
    """
    # Log raw sanitized exception for developer diagnostics
    clean_err_str = sanitize_secrets(str(err))
    logger.error("Application error [%s]: %s", type(err).__name__, clean_err_str)

    # 1. URL & Video ID
    if isinstance(err, InvalidYouTubeURLError):
        return (
            "Invalid YouTube URL",
            f"The provided link is not a recognized YouTube URL. {clean_err_str}",
        )
    if isinstance(err, InvalidVideoIDError):
        return (
            "Invalid Video ID",
            f"The extracted video ID is invalid. {clean_err_str}",
        )

    # 2. Transcripts
    if isinstance(err, CaptionsDisabledError):
        return (
            "Subtitles Disabled",
            "Subtitles/closed captions have been disabled by the video owner. "
            "Zero-download processing requires available caption tracks.",
        )
    if isinstance(err, EnglishTranscriptNotFoundError):
        return (
            "English Captions Unavailable",
            "English captions are unavailable for this video. "
            "Native YouTube subtitles could not be found.",
        )
    if isinstance(err, VideoUnavailableError):
        return (
            "Video Unavailable",
            "The YouTube video is private, removed, or geographically restricted.",
        )
    if isinstance(err, EmptyTranscriptError):
        return (
            "Empty Transcript",
            "The video transcript contains no readable speech or text content after cleaning.",
        )
    if isinstance(err, TranscriptTooLongError):
        return (
            "Transcript Exceeds Limit",
            f"The transcript exceeds maximum allowed processing limits: {clean_err_str}",
        )
    if isinstance(err, TranscriptAPIError):
        return (
            "Transcript Service Error",
            f"Failed to query YouTube closed captions service. {clean_err_str}",
        )
    if isinstance(err, TranscriptError):
        return (
            "Transcript Error",
            f"An error occurred while retrieving subtitles: {clean_err_str}",
        )

    # 3. Chunking
    if isinstance(err, ChunkingError):
        return (
            "Context Chunking Error",
            f"Failed to partition transcript into context chunks. {clean_err_str}",
        )

    # 4. LLM & NIM
    if isinstance(err, LLMConfigurationError):
        return (
            "NVIDIA Configuration Missing",
            "NVIDIA API Key was not found. Please set `NVIDIA_API_KEY` in your `.env` file.",
        )
    if isinstance(err, LLMAuthenticationError):
        return (
            "NVIDIA Authentication Failure (401)",
            "The provided NVIDIA API Key was rejected as unauthorized. "
            "Please verify your `NVIDIA_API_KEY` in `.env`.",
        )
    if isinstance(err, LLMRateLimitError):
        return (
            "NVIDIA Rate Limit / Quota Exceeded (429)",
            "Your NVIDIA NIM request limit or quota cap has been reached. "
            "Please pause a moment before retrying.",
        )
    if isinstance(err, LLMTimeoutError):
        return (
            "LLM Request Timeout",
            "The NVIDIA NIM request timed out. Please check your network connection "
            "or increase `LLM_TIMEOUT_SECONDS`.",
        )
    if isinstance(err, LLMConnectionError):
        return (
            "LLM Connection Failure",
            "Could not connect to the NVIDIA NIM endpoint. Please check your internet connectivity.",
        )
    if isinstance(err, LLMContextOverflowError):
        return (
            "Model Context Overflow",
            "The input prompt exceeded the model's maximum token context window. "
            "Try reducing `Max Words per Chunk` in the sidebar.",
        )
    if isinstance(err, EmptyLLMResponseError):
        return (
            "Empty LLM Response",
            "The NVIDIA NIM service returned an empty or blank response. Please retry.",
        )
    if isinstance(err, LLMError):
        return (
            "LLM Generation Error",
            f"An error occurred during AI inference: {clean_err_str}",
        )

    # 5. Output Validation
    if isinstance(err, MissingRequiredSectionError):
        return (
            "Incomplete Notes Generated",
            f"The model response failed structural validation. {clean_err_str}",
        )
    if isinstance(err, MarkdownValidationError):
        return (
            "Malformed Markdown Output",
            f"The generated notes could not be verified against the required schema: {clean_err_str}",
        )

    # 6. Fallback Generic
    if isinstance(err, AppError):
        return ("Application Error", clean_err_str)

    return (
        "Unexpected System Error",
        f"An unexpected failure occurred: {clean_err_str}. Please check application logs.",
    )

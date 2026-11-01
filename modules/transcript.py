"""Zero-download YouTube transcript acquisition module and provider abstraction.

Provides the TranscriptProvider abstract interface with YouTubeTranscriptProvider
as the primary zero-download implementation and WhisperFallbackProvider as an
extensible, zero-download architecture for explicitly supplied local audio.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
)

from modules.url_parser import VIDEO_ID_REGEX

logger = logging.getLogger(__name__)

# Preferred English language codes in order of priority
PREFERRED_ENGLISH_LANGUAGES = [
    "en",
    "en-US",
    "en-GB",
    "en-CA",
    "en-AU",
    "en-IN",
    "en-IE",
    "en-NZ",
]


from modules.exceptions import (
    CaptionsDisabledError,
    EnglishTranscriptNotFoundError,
    InvalidVideoIDError,
    TranscriptAPIError,
    VideoUnavailableError,
)



# ---------------------------------------------------------------------------
# Abstract Transcript Provider
# ---------------------------------------------------------------------------

class TranscriptProvider(ABC):
    """Abstract Base Class defining the contract for subtitle and transcript providers."""

    @abstractmethod
    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieve transcript segments for a given video ID.

        Args:
            video_id: 11-character YouTube video ID.

        Returns:
            List of dictionaries:
                [
                    {
                        "text": str,
                        "start": float,
                        "duration": float
                    },
                    ...
                ]
        """
        pass


# ---------------------------------------------------------------------------
# Primary Provider: YouTube Closed Captions (Zero-Download)
# ---------------------------------------------------------------------------

def _get_api_client():
    """Return a callable YouTubeTranscriptApi instance or class for API backwards-compatibility."""
    try:
        return YouTubeTranscriptApi()
    except TypeError:
        return YouTubeTranscriptApi


def _list_transcripts(client, video_id: str):
    """List available transcripts using the appropriate API method across versions."""
    if hasattr(client, "list"):
        return client.list(video_id)
    if hasattr(client, "list_transcripts"):
        return client.list_transcripts(video_id)
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        return YouTubeTranscriptApi.list_transcripts(video_id)
    raise AttributeError("No supported transcript listing method found on YouTubeTranscriptApi.")


class YouTubeTranscriptProvider(TranscriptProvider):
    """Primary zero-download transcript provider querying YouTube closed caption tracks."""

    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieve the English transcript for a given YouTube video ID without media downloading.

        Args:
            video_id: 11-character YouTube video ID.

        Returns:
            Structured list of caption segment dictionaries.

        Raises:
            InvalidVideoIDError: If the video ID is malformed.
            CaptionsDisabledError: If captions are disabled on the video.
            VideoUnavailableError: If the video is private or removed.
            EnglishTranscriptNotFoundError: If no English transcript track exists.
            TranscriptAPIError: If the network request fails.
        """
        if not video_id or not isinstance(video_id, str):
            raise InvalidVideoIDError("Video ID must be a non-empty string.")

        cleaned_id = video_id.strip()
        if len(cleaned_id) != 11 or not VIDEO_ID_REGEX.match(cleaned_id):
            raise InvalidVideoIDError(
                f"Invalid video ID format '{cleaned_id}'. Expected exactly 11 characters."
            )

        logger.info("Requesting transcript for video ID: %s via YouTubeTranscriptProvider", cleaned_id)
        client = _get_api_client()

        try:
            transcript_list = _list_transcripts(client, cleaned_id)
        except TranscriptsDisabled as e:
            logger.warning("Subtitles are disabled for video: %s", cleaned_id)
            raise CaptionsDisabledError(
                f"Subtitles/closed captions are disabled for video ID '{cleaned_id}'."
            ) from e
        except VideoUnavailable as e:
            logger.warning("Video unavailable for ID: %s", cleaned_id)
            raise VideoUnavailableError(
                f"The video with ID '{cleaned_id}' is unavailable, private, or has been removed."
            ) from e
        except YouTubeRequestFailed as e:
            logger.error("YouTube request failed for video ID %s: %s", cleaned_id, e)
            raise TranscriptAPIError(
                f"YouTube request failed while querying transcripts: {e}"
            ) from e
        except Exception as e:
            logger.exception("Unexpected error listing transcripts for video ID: %s", cleaned_id)
            raise TranscriptAPIError(f"Failed to query transcript list: {e}") from e

        # 1. Prefer manually created English transcripts
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(PREFERRED_ENGLISH_LANGUAGES)
            logger.info("Found manually created English transcript for: %s", cleaned_id)
        except NoTranscriptFound:
            logger.info("No manually created English transcript found. Checking auto-generated.")

        # 2. Fall back to auto-generated English transcripts
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(PREFERRED_ENGLISH_LANGUAGES)
                logger.info("Found auto-generated English transcript for: %s", cleaned_id)
            except NoTranscriptFound:
                logger.info("No auto-generated English transcript found.")

        # 3. Final fallback: find any matching English code directly
        if transcript is None:
            try:
                transcript = transcript_list.find_transcript(PREFERRED_ENGLISH_LANGUAGES)
            except NoTranscriptFound:
                pass

        if transcript is None:
            available_langs = [f"{t.language} ({t.language_code})" for t in transcript_list]
            raise EnglishTranscriptNotFoundError(
                f"No English transcript available for video ID '{cleaned_id}'. "
                f"Available languages: {', '.join(available_langs) if available_langs else 'None'}."
            )

        # Fetch the caption segments
        try:
            raw_result = transcript.fetch()
        except CouldNotRetrieveTranscript as e:
            logger.error("Could not retrieve transcript data for %s: %s", cleaned_id, e)
            raise TranscriptAPIError(f"Could not retrieve transcript segments: {e}") from e
        except Exception as e:
            logger.exception("Failed to fetch transcript segments for: %s", cleaned_id)
            raise TranscriptAPIError(f"Failed to fetch transcript content: {e}") from e

        # Handle FetchedTranscript objects or raw dictionaries
        if hasattr(raw_result, "to_raw_data"):
            raw_items = raw_result.to_raw_data()
        else:
            raw_items = raw_result

        # Format and clean segments
        formatted_segments: List[Dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict):
                text = item.get("text", "")
                start = item.get("start", 0.0)
                duration = item.get("duration", 0.0)
            else:
                text = getattr(item, "text", "")
                start = getattr(item, "start", 0.0)
                duration = getattr(item, "duration", 0.0)

            formatted_segments.append(
                {
                    "text": str(text).strip(),
                    "start": round(float(start), 2),
                    "duration": round(float(duration), 2),
                }
            )

        logger.info(
            "Successfully fetched %d transcript segments for video ID: %s",
            len(formatted_segments),
            cleaned_id,
        )
        return formatted_segments


# ---------------------------------------------------------------------------
# Optional Extension Architecture: Whisper Fallback Provider
# ---------------------------------------------------------------------------

class WhisperFallbackProvider(TranscriptProvider):
    """Optional Whisper fallback architecture for user-supplied audio files.

    Strict Architectural Invariants:
    1. NEVER automatically downloads YouTube video or audio streams.
    2. Zero unauthorized third-party media stream scrapers (no yt-dlp, no pytube).
    3. Designed to receive an explicitly provided local audio file if captions are absent.
    """

    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Reject automated video downloading to preserve the zero-download architecture.

        Raises:
            EnglishTranscriptNotFoundError: Informs caller that native captions were absent
                and automated downloading is intentionally prohibited.
        """
        raise EnglishTranscriptNotFoundError(
            f"English captions are unavailable for video ID '{video_id}'. "
            "Automated YouTube audio/video downloading is prohibited to preserve zero-download architecture. "
            "To use Whisper transcription, explicitly supply a local audio file via transcribe_local_audio()."
        )

    def transcribe_local_audio(
        self,
        audio_file_path: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extension hook: transcribe an explicitly supplied local audio file.

        Can be wired to Groq's high-speed Whisper endpoint (e.g., whisper-large-v3
        on console.groq.com) or a local faster-whisper instance.

        Args:
            audio_file_path: Absolute path to local audio file (.mp3, .wav, .m4a).
            api_key: Optional API key (e.g. GROQ_API_KEY).
            base_url: Optional API endpoint URL.

        Returns:
            List of transcript segment dictionaries with text and timestamps.

        Raises:
            FileNotFoundError: If the specified local audio file does not exist.
            NotImplementedError: Placeholder for future local audio integration.
        """
        import os
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Local audio file not found: {audio_file_path}")

        # Documented architectural extension:
        # In a future release, this method can call Groq's OpenAI-compatible
        # client: client.audio.transcriptions.create(model="whisper-large-v3", file=...)
        raise NotImplementedError(
            "Local audio transcription via Groq / Whisper is architecturally defined. "
            "Configure GROQ_API_KEY and supply local audio to activate."
        )


# Default convenience wrapper maintaining backwards compatibility
def get_transcript(video_id: str) -> List[Dict[str, Any]]:
    """Retrieve transcript segments using the default zero-download YouTube provider."""
    provider = YouTubeTranscriptProvider()
    return provider.get_transcript(video_id)

"""Unit tests for modules.transcript using mocks and provider validation."""

import unittest
from unittest.mock import MagicMock, patch

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

from modules.transcript import (
    CaptionsDisabledError,
    EnglishTranscriptNotFoundError,
    InvalidVideoIDError,
    TranscriptAPIError,
    TranscriptProvider,
    VideoUnavailableError,
    WhisperFallbackProvider,
    YouTubeTranscriptProvider,
    get_transcript,
)


class TestTranscriptAcquisition(unittest.TestCase):
    """Test suite verifying zero-download transcript retrieval and error handling."""

    def setUp(self):
        self.sample_raw_segments = [
            {"text": "Hello world", "start": 0.0, "duration": 2.5},
            {"text": "Welcome to the video", "start": 2.5, "duration": 3.0},
        ]

    def test_transcript_provider_abstraction(self):
        """Test that TranscriptProvider is an ABC and cannot be directly instantiated."""
        with self.assertRaises(TypeError):
            TranscriptProvider()  # Can't instantiate abstract class

        self.assertTrue(issubclass(YouTubeTranscriptProvider, TranscriptProvider))
        self.assertTrue(issubclass(WhisperFallbackProvider, TranscriptProvider))

    def test_whisper_fallback_rejects_automated_media_downloads(self):
        """Test that Whisper fallback intentionally refuses automated YouTube media downloading."""
        whisper_provider = WhisperFallbackProvider()
        with self.assertRaises(EnglishTranscriptNotFoundError) as ctx:
            whisper_provider.get_transcript("dQw4w9WgXcQ")
        self.assertIn("zero-download architecture", str(ctx.exception))

    def test_invalid_video_id(self):
        """Test rejection of empty or malformed video IDs."""
        with self.assertRaises(InvalidVideoIDError):
            get_transcript("")
        with self.assertRaises(InvalidVideoIDError):
            get_transcript("short")
        with self.assertRaises(InvalidVideoIDError):
            get_transcript("invalid!char")

    @patch("modules.transcript._list_transcripts")
    def test_successful_manual_transcript_retrieval(self, mock_list_transcripts):
        """Test happy path where manually created English transcript is found."""
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = self.sample_raw_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.return_value = mock_transcript
        mock_list_transcripts.return_value = mock_transcript_list

        provider = YouTubeTranscriptProvider()
        segments = provider.get_transcript("dQw4w9WgXcQ")

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "Hello world")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["duration"], 2.5)
        mock_transcript_list.find_manually_created_transcript.assert_called_once()

    @patch("modules.transcript._list_transcripts")
    def test_fallback_to_generated_transcript(self, mock_list_transcripts):
        """Test fallback when manual transcript is missing but generated is available."""
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = self.sample_raw_segments

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.side_effect = NoTranscriptFound("dQw4w9WgXcQ", ["en"], [])
        mock_transcript_list.find_generated_transcript.return_value = mock_transcript
        mock_list_transcripts.return_value = mock_transcript_list

        segments = get_transcript("dQw4w9WgXcQ")

        self.assertEqual(len(segments), 2)
        mock_transcript_list.find_generated_transcript.assert_called_once()

    @patch("modules.transcript._list_transcripts")
    def test_captions_disabled(self, mock_list_transcripts):
        """Test error when captions are disabled for the video."""
        mock_list_transcripts.side_effect = TranscriptsDisabled("dQw4w9WgXcQ")

        with self.assertRaises(CaptionsDisabledError):
            get_transcript("dQw4w9WgXcQ")

    @patch("modules.transcript._list_transcripts")
    def test_video_unavailable(self, mock_list_transcripts):
        """Test error when video is unavailable or removed."""
        mock_list_transcripts.side_effect = VideoUnavailable("dQw4w9WgXcQ")

        with self.assertRaises(VideoUnavailableError):
            get_transcript("dQw4w9WgXcQ")

    @patch("modules.transcript._list_transcripts")
    def test_english_transcript_not_found(self, mock_list_transcripts):
        """Test error when no English transcripts (manual or generated) are available."""
        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.side_effect = NoTranscriptFound("dQw4w9WgXcQ", ["en"], [])
        mock_transcript_list.find_generated_transcript.side_effect = NoTranscriptFound("dQw4w9WgXcQ", ["en"], [])
        mock_transcript_list.find_transcript.side_effect = NoTranscriptFound("dQw4w9WgXcQ", ["en"], [])
        mock_transcript_list.__iter__.return_value = []
        mock_list_transcripts.return_value = mock_transcript_list

        with self.assertRaises(EnglishTranscriptNotFoundError):
            get_transcript("dQw4w9WgXcQ")

    @patch("modules.transcript._list_transcripts")
    def test_api_network_error(self, mock_list_transcripts):
        """Test error when YouTube request fails."""
        mock_list_transcripts.side_effect = YouTubeRequestFailed("dQw4w9WgXcQ", Exception("Connection dropped"))

        with self.assertRaises(TranscriptAPIError):
            get_transcript("dQw4w9WgXcQ")


if __name__ == "__main__":
    unittest.main()

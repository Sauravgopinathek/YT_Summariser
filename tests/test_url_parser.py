"""Unit tests for modules.url_parser."""

import unittest
from modules.url_parser import (
    InvalidYouTubeURLError,
    extract_video_id,
    validate_youtube_url,
)


class TestYouTubeURLParser(unittest.TestCase):
    """Test suite covering various YouTube URL patterns and edge cases."""

    def test_normal_youtube_url(self):
        """Test standard https://www.youtube.com/watch?v=VIDEO_ID format."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
        self.assertTrue(validate_youtube_url(url))

        # Without www
        url_no_www = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url_no_www), "dQw4w9WgXcQ")

    def test_shortened_url(self):
        """Test shortened https://youtu.be/VIDEO_ID format."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
        self.assertTrue(validate_youtube_url(url))

    def test_embed_url(self):
        """Test embed https://www.youtube.com/embed/VIDEO_ID format."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
        self.assertTrue(validate_youtube_url(url))

    def test_url_with_timestamp_and_extra_parameters(self):
        """Test URLs with query parameters like ?t=..., &feature=..., ?si=..."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45s",
            "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ&si=12345",
            "https://youtu.be/dQw4w9WgXcQ?t=120",
            "https://youtu.be/dQw4w9WgXcQ?si=abcdef123456",
            "https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1&si=xyz789",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
                self.assertTrue(validate_youtube_url(url))

    def test_invalid_url_domain(self):
        """Test malformed URLs or non-YouTube domains."""
        invalid_urls = [
            "https://www.google.com",
            "https://vimeo.com/123456789",
            "not-a-valid-url",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(InvalidYouTubeURLError):
                    extract_video_id(url)
                self.assertFalse(validate_youtube_url(url))

    def test_empty_or_whitespace_url(self):
        """Test empty strings or whitespace input."""
        for empty_val in ["", "   ", "\t\n"]:
            with self.subTest(empty_val=empty_val):
                with self.assertRaises(InvalidYouTubeURLError):
                    extract_video_id(empty_val)
                self.assertFalse(validate_youtube_url(empty_val))

    def test_missing_video_id(self):
        """Test YouTube URLs missing the video ID."""
        missing_id_urls = [
            "https://www.youtube.com/watch",
            "https://www.youtube.com/watch?",
            "https://www.youtube.com/watch?v=",
            "https://youtu.be/",
            "https://www.youtube.com/embed/",
        ]
        for url in missing_id_urls:
            with self.subTest(url=url):
                with self.assertRaises(InvalidYouTubeURLError):
                    extract_video_id(url)
                self.assertFalse(validate_youtube_url(url))

    def test_incorrect_id_length(self):
        """Test URLs with video IDs shorter or longer than 11 characters."""
        incorrect_length_urls = [
            "https://www.youtube.com/watch?v=shortID",  # 7 chars
            "https://www.youtube.com/watch?v=toolongvideoidentifier123",  # 24 chars
            "https://youtu.be/1234567890",  # 10 chars
            "https://youtu.be/123456789012",  # 12 chars
        ]
        for url in incorrect_length_urls:
            with self.subTest(url=url):
                with self.assertRaises(InvalidYouTubeURLError) as context:
                    extract_video_id(url)
                self.assertIn("Invalid video ID length", str(context.exception))
                self.assertFalse(validate_youtube_url(url))


if __name__ == "__main__":
    unittest.main()

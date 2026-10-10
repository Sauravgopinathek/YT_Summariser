"""Unit tests for modules.preprocessing."""

import unittest
from modules.preprocessing import (
    clean_transcript,
    get_transcript_statistics,
    normalize_text,
)


class TestTranscriptPreprocessing(unittest.TestCase):
    """Test suite for subtitle cleaning and text preprocessing."""

    def test_sample_subtitle_fragments_concatenation(self):
        """Test the example given in requirements: merging fragmented sentences."""
        segments = [
            {"text": "Today we are going to", "start": 0, "duration": 2},
            {"text": "learn about machine learning.", "start": 2, "duration": 2},
        ]
        result = clean_transcript(segments)
        expected_text = "Today we are going to learn about machine learning."
        self.assertEqual(result["clean_text"], expected_text)
        self.assertEqual(result["word_count"], 9)
        self.assertEqual(result["segment_count"], 2)
        self.assertEqual(result["character_count"], len(expected_text))

    def test_html_entity_decoding(self):
        """Test unescaping of HTML entities in subtitle segments."""
        raw_text = "It&#39;s an AI &amp; ML algorithm &quot;test&quot; &lt;tag&gt;"
        cleaned = normalize_text(raw_text)
        self.assertEqual(cleaned, "It's an AI & ML algorithm \"test\" <tag>")

    def test_artifact_removal(self):
        """Test stripping of subtitle audio cues and music notes."""
        raw_texts = [
            "[Music] Welcome to the tutorial [Applause]",
            "Now let us continue (laughter) with the demo ♪",
            "[inaudible] [cheering] ♫ Thanks everyone",
        ]
        segments = [{"text": t, "start": i, "duration": 2} for i, t in enumerate(raw_texts)]
        result = clean_transcript(segments)
        expected_text = "Welcome to the tutorial Now let us continue with the demo Thanks everyone"
        self.assertEqual(result["clean_text"], expected_text)

    def test_preservation_of_technical_terms_and_formulas(self):
        """Test preservation of technical names, code symbols, formulas, and numbers."""
        tech_text = (
            "Deploying PyTorch 2.4 and Python 3.13 on NVIDIA H100 GPU. "
            "Cost is $2,500 with 99.9% uptime. Formula: E = mc^2, Complexity: O(N log N)."
        )
        segments = [{"text": tech_text, "start": 0, "duration": 5}]
        result = clean_transcript(segments)
        self.assertEqual(result["clean_text"], tech_text)
        self.assertIn("PyTorch 2.4", result["clean_text"])
        self.assertIn("NVIDIA H100 GPU", result["clean_text"])
        self.assertIn("$2,500", result["clean_text"])
        self.assertIn("E = mc^2", result["clean_text"])
        self.assertIn("O(N log N)", result["clean_text"])

    def test_whitespace_and_punctuation_spacing(self):
        """Test collapsing extra whitespace, newlines, and orphan punctuation spaces."""
        segments = [
            {"text": "First point   \n\t  is here", "start": 0, "duration": 1},
            {"text": " , and the second point .", "start": 1, "duration": 1},
        ]
        result = clean_transcript(segments)
        self.assertEqual(result["clean_text"], "First point is here, and the second point.")

    def test_empty_and_edge_cases(self):
        """Test handling of empty lists, None, and all-artifact segments."""
        self.assertEqual(clean_transcript([])["clean_text"], "")
        self.assertEqual(clean_transcript([])["word_count"], 0)
        self.assertEqual(clean_transcript(None)["clean_text"], "")

        # Only music artifacts
        artifact_only = [{"text": "[Music]", "start": 0, "duration": 1}, {"text": "♪ ♫", "start": 1, "duration": 1}]
        result = clean_transcript(artifact_only)
        self.assertEqual(result["clean_text"], "")
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["segment_count"], 2)

    def test_get_transcript_statistics(self):
        """Test statistics calculation directly."""
        stats = get_transcript_statistics("One two three four five.", segment_count=10)
        self.assertEqual(stats["word_count"], 5)
        self.assertEqual(stats["character_count"], 24)
        self.assertEqual(stats["segment_count"], 10)


if __name__ == "__main__":
    unittest.main()

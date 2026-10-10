"""Unit tests for modules.summarizer."""

import unittest
from unittest.mock import MagicMock

from modules.llm import NVIDIAClient
from modules.summarizer import generate_notes, parse_markdown_notes


class TestSummarizerEngine(unittest.TestCase):
    """Test suite for direct & hierarchical summarization and Markdown parsing."""

    def setUp(self):
        self.sample_markdown = (
            "# Executive Summary\n"
            "This video explores convolutional neural networks and spatial filters.\n\n"
            "# Key Concepts\n"
            "- **Kernels**: 3x3 matrices used to detect edges.\n"
            "- **Pooling**: Reduces spatial dimensions while preserving dominant features.\n\n"
            "# Action Items\n"
            "- Use 3x3 kernels for standard feature extraction.\n"
            "- Apply batch normalization after convolutional layers."
        )

    def test_parse_markdown_notes(self):
        """Test parsing of well-formed markdown notes."""
        parsed = parse_markdown_notes(self.sample_markdown)

        self.assertIn("convolutional neural networks", parsed["executive_summary"])
        self.assertEqual(len(parsed["key_concepts"]), 2)
        self.assertIn("**Kernels**", parsed["key_concepts"][0])
        self.assertEqual(len(parsed["action_items"]), 2)
        self.assertIn("batch normalization", parsed["action_items"][1])

    def test_parse_markdown_notes_fallback_without_headers(self):
        """Test graceful fallback when output lacks markdown headers."""
        raw_text = "This is a direct text summary without any explicit markdown headers."
        parsed = parse_markdown_notes(raw_text)

        self.assertEqual(parsed["executive_summary"], raw_text)
        self.assertEqual(parsed["key_concepts"], [])
        self.assertEqual(parsed["action_items"], [])

    def test_empty_transcript_raises_value_error(self):
        """Test that empty or whitespace transcript raises ValueError."""
        with self.assertRaises(ValueError):
            generate_notes("")
        with self.assertRaises(ValueError):
            generate_notes("    ")

    def test_direct_summarization_strategy(self):
        """Test that a short transcript triggers the direct strategy (1 LLM call)."""
        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = self.sample_markdown

        short_transcript = "Short lecture on convolution kernels and pooling operations."
        result = generate_notes(
            transcript=short_transcript,
            client=mock_client,
            max_words_per_chunk=100,
        )

        self.assertEqual(result["metadata"]["strategy"], "direct")
        self.assertEqual(result["metadata"]["llm_call_count"], 1)
        self.assertEqual(result["metadata"]["chunk_count"], 1)
        self.assertEqual(len(result["key_concepts"]), 2)
        self.assertEqual(len(result["action_items"]), 2)
        self.assertIn("# Executive Summary", result["markdown"])
        mock_client.generate.assert_called_once()

    def test_map_reduce_summarization_strategy(self):
        """Test that a long transcript spanning multiple chunks triggers map-reduce."""
        mock_client = MagicMock(spec=NVIDIAClient)
        # 3 calls: chunk 1 map, chunk 2 map, reduce
        mock_client.generate.side_effect = [
            "Section 1 summary: Introduces convolution.",
            "Section 2 summary: Discusses pooling layers.",
            self.sample_markdown,
        ]

        # Generate a transcript that will be split into 2 chunks with max_words_per_chunk=15
        long_transcript = (
            "Sentence one describes the convolution operator and spatial kernels in vision models. "
            "Sentence two explains pooling layers, downsampling mechanisms, and feature hierarchy."
        )

        result = generate_notes(
            transcript=long_transcript,
            client=mock_client,
            max_words_per_chunk=15,
            overlap_words=3,
        )

        self.assertEqual(result["metadata"]["strategy"], "map_reduce")
        self.assertGreaterEqual(result["metadata"]["chunk_count"], 2)
        self.assertEqual(
            result["metadata"]["llm_call_count"],
            result["metadata"]["chunk_count"] + 1,
        )
        self.assertIn("convolutional neural networks", result["executive_summary"])
        self.assertGreaterEqual(result["metadata"]["processing_time_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()

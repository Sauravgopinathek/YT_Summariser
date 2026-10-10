"""End-to-End Pipeline Integration Test.

Verifies the entire production pipeline:
YouTube URL
→ Video ID
→ Transcript Retrieval
→ Clean Text
→ Chunks
→ NVIDIA LLM
→ Structured Notes
→ Markdown Notes
→ Faithfulness Validation
"""

import unittest
from unittest.mock import MagicMock

from modules.chunking import chunk_transcript
from modules.formatter import format_as_markdown, validate_markdown_notes
from modules.llm import NVIDIAClient
from modules.preprocessing import clean_transcript
from modules.summarizer import generate_notes
from modules.transcript import YouTubeTranscriptProvider
from modules.url_parser import extract_video_id
from modules.validation import validate_summary_faithfulness


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration test validating the entire summarization pipeline."""

    def test_full_pipeline_with_mock_llm(self):
        """Test full pipeline end-to-end from YouTube URL to validated Markdown notes."""
        raw_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=shared"

        # 1. URL Parsing & Video ID Extraction
        video_id = extract_video_id(raw_url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")

        # 2. Transcript Acquisition (Simulated raw subtitle segments)
        raw_segments = [
            {"text": "Welcome to this lecture on Deep Neural Networks.", "start": 0.0, "duration": 2.5},
            {"text": "[Music] Today we cover backpropagation and gradient descent.", "start": 2.5, "duration": 3.0},
            {"text": "The weight update formulation is W = W - alpha * dL/dW.", "start": 5.5, "duration": 3.5},
            {"text": "Always normalize your inputs to zero mean before training.", "start": 9.0, "duration": 3.0},
        ]

        # 3. Preprocessing
        preprocessed = clean_transcript(raw_segments)
        clean_text = preprocessed["clean_text"]
        self.assertIn("Deep Neural Networks", clean_text)
        self.assertIn("W = W - alpha * dL/dW", clean_text)
        self.assertNotIn("[Music]", clean_text)
        self.assertGreater(preprocessed["word_count"], 15)

        # 4. Chunking
        chunks = chunk_transcript(clean_text, max_words=50, overlap_words=10)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertGreater(chunk["word_count"], 0)
            self.assertIn("chunk_id", chunk)

        # 5. LLM Client & Notes Generation
        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = (
            "# Executive Summary\n"
            "This lecture introduces deep neural networks, focusing on backpropagation "
            "and the fundamental gradient descent weight update formulation.\n\n"
            "# Key Concepts\n"
            "- Backpropagation and Gradient Descent: Core optimization methods.\n"
            "- Weight Update Formula: W = W - alpha * dL/dW.\n\n"
            "# Action Items\n"
            "- Normalize input features to zero mean before training."
        )

        notes = generate_notes(
            transcript=clean_text,
            client=mock_client,
            max_words_per_chunk=50,
            overlap_words=10,
        )

        self.assertIn("executive_summary", notes)
        self.assertIn("key_concepts", notes)
        self.assertIn("action_items", notes)
        self.assertIn("metadata", notes)

        # 6. Markdown Formatting & Validation
        formatted_md = format_as_markdown(notes)
        is_valid = validate_markdown_notes(formatted_md)
        self.assertTrue(is_valid, "Markdown must pass structural validation")
        self.assertIn("# Executive Summary", formatted_md)
        self.assertIn("# Key Concepts", formatted_md)
        self.assertIn("# Action Items", formatted_md)


        # 7. Grounding & Faithfulness Validation
        faithfulness = validate_summary_faithfulness(clean_text, formatted_md)
        self.assertTrue(faithfulness["valid"])
        self.assertGreaterEqual(faithfulness["faithfulness_score"], 0.70)
        self.assertIsInstance(faithfulness["coverage_score"], float)


if __name__ == "__main__":
    unittest.main()

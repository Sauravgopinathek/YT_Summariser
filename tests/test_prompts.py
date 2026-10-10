"""Unit tests for prompts loader and prompt template structures."""

from pathlib import Path
import unittest

from prompts.loader import (
    get_summary_user_prompt,
    get_system_prompt,
    load_prompt_file,
)


class TestPromptTemplates(unittest.TestCase):
    """Test suite verifying prompt template loading, placeholders, and structure."""

    def test_system_prompt_loading_and_rules(self):
        """Test system prompt loads correctly and contains essential guardrail rules."""
        sys_prompt = get_system_prompt()
        self.assertIsInstance(sys_prompt, str)
        self.assertGreater(len(sys_prompt), 100)
        self.assertIn("STRICT GROUNDING", sys_prompt)
        self.assertIn("PRESERVATION OF TECHNICAL INTEGRITY", sys_prompt)
        self.assertIn("NO HALLUCINATION", sys_prompt)
        self.assertIn("Markdown", sys_prompt)

    def test_summary_prompt_formatting_and_sections(self):
        """Test summary user prompt loads and injects transcript into placeholder."""
        sample_transcript = "This is a lecture on deep learning and convolution operations."
        formatted_prompt = get_summary_user_prompt(sample_transcript)

        self.assertIn(sample_transcript, formatted_prompt)
        self.assertNotIn("{transcript}", formatted_prompt)

        # Check required mandatory headers
        self.assertIn("# Executive Summary", formatted_prompt)
        self.assertIn("# Key Concepts", formatted_prompt)
        self.assertIn("# Action Items", formatted_prompt)

        # Check explicit fallback requirement
        self.assertIn("No explicit action items were identified in the transcript.", formatted_prompt)

    def test_missing_prompt_file_raises_error(self):
        """Test load_prompt_file raises FileNotFoundError for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_prompt_file(Path("non_existent_prompt_template.txt"))


if __name__ == "__main__":
    unittest.main()

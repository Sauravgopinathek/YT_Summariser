"""Unit tests for modules.validation."""

import unittest
from unittest.mock import MagicMock

from modules.llm import NVIDIAClient
from modules.validation import (
    extract_key_terminology,
    extract_numerical_claims,
    validate_summary_faithfulness,
    verify_faithfulness_with_llm,
)


class TestGroundingValidation(unittest.TestCase):
    """Test suite for faithfulness validation, hallucination detection, and term extraction."""

    def setUp(self):
        self.sample_transcript = (
            "In this tutorial on PyTorch 2.4 and CUDA, we optimize deep neural networks. "
            "We achieve an accuracy of 94.5% using the Adam optimizer with a learning rate of 0.001. "
            "Training took 10 epochs on an NVIDIA H100 GPU."
        )

        self.faithful_summary = (
            "# Executive Summary\n"
            "This tutorial demonstrates optimizing neural networks with PyTorch 2.4 and CUDA on an NVIDIA H100 GPU.\n\n"
            "# Key Concepts\n"
            "- **Adam Optimizer**: Used with a learning rate of 0.001.\n"
            "- **Performance**: Reached 94.5% accuracy over 10 epochs.\n\n"
            "# Action Items\n"
            "- Configure PyTorch for CUDA acceleration.\n"
            "- Set learning rate to 0.001."
        )

        self.hallucinated_summary = (
            "# Executive Summary\n"
            "This tutorial claims PyTorch achieved $25,000 in revenue and 99.9% market dominance.\n\n"
            "# Key Concepts\n"
            "- **Profit**: $25,000.\n\n"
            "# Action Items\n"
            "- Invest in 2035."
        )

    def test_extract_numerical_claims(self):
        """Test extraction of percentages, currency, and numbers."""
        claims = extract_numerical_claims("Cost is $1,500 with 98.5% uptime in year 2026.")
        self.assertIn("$1,500", claims)
        self.assertIn("98.5%", claims)
        self.assertIn("2026", claims)

    def test_extract_key_terminology(self):
        """Test extraction of acronyms and camelCase technical terms."""
        terms = extract_key_terminology("Using PyTorch, CUDA, and CNN architectures on GPU.")
        self.assertIn("PyTorch", terms)
        self.assertIn("CUDA", terms)
        self.assertIn("CNN", terms)
        self.assertIn("GPU", terms)

    def test_faithful_summary_scores_high(self):
        """Test grounded faithful summary receives high score and no warnings."""
        result = validate_summary_faithfulness(self.sample_transcript, self.faithful_summary)
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["faithfulness_score"], 0.7)
        self.assertEqual(len(result["unsupported_numbers"]), 0)
        self.assertIn("disclaimer", result)

    def test_unsupported_numbers_detection(self):
        """Test that numbers not present in source transcript trigger warnings."""
        result = validate_summary_faithfulness(self.sample_transcript, self.hallucinated_summary)
        self.assertGreater(len(result["unsupported_numbers"]), 0)
        # Should detect $25,000 and 99.9% as unsupported
        self.assertTrue(any("$25,000" in n or "25,000" in n for n in result["unsupported_numbers"]))
        self.assertGreater(len(result["warnings"]), 0)

    def test_empty_summary_validation(self):
        """Test that empty summary returns valid=False and score 0.0."""
        result = validate_summary_faithfulness(self.sample_transcript, "")
        self.assertFalse(result["valid"])
        self.assertEqual(result["faithfulness_score"], 0.0)

    def test_optional_llm_verification_pass(self):
        """Test LLM-based verification pass with mocked client."""
        mock_client = MagicMock(spec=NVIDIAClient)
        mock_client.generate.return_value = "VERDICT: FAITHFUL\nAll statements are supported by the transcript."

        audit = verify_faithfulness_with_llm(
            transcript=self.sample_transcript,
            summary_markdown=self.faithful_summary,
            client=mock_client,
        )

        self.assertTrue(audit["llm_verified"])
        self.assertTrue(audit["is_faithful"])
        self.assertIn("VERDICT: FAITHFUL", audit["analysis"])


if __name__ == "__main__":
    unittest.main()

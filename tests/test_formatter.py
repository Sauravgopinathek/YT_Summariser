"""Unit tests for modules.formatter."""

import unittest

from modules.formatter import (
    MarkdownValidationError,
    format_as_markdown,
    strip_emojis,
    validate_markdown_notes,
)


class TestMarkdownFormatter(unittest.TestCase):
    """Test suite for Markdown formatting, emoji removal, and structural validation."""

    def test_format_as_markdown_standard(self):
        """Test standard conversion from structured dictionary to clean Markdown."""
        data = {
            "executive_summary": "This talk explains neural network backpropagation and gradient descent.",
            "key_concepts": [
                "**Gradient Vector**: Vector of partial derivatives with respect to weights.",
                "**Learning Rate**: Step size scaling factor in optimization.",
            ],
            "action_items": [
                "Normalize inputs to zero mean before training.",
                "Log epoch losses to TensorBoard.",
            ],
        }

        md = format_as_markdown(data)
        self.assertIn("# Executive Summary\n\nThis talk explains", md)
        self.assertIn("# Key Concepts\n\n- **Gradient Vector**", md)
        self.assertIn("# Action Items\n\n- Normalize inputs", md)
        self.assertTrue(validate_markdown_notes(md))

    def test_validation_passes_valid_markdown(self):
        """Test validate_markdown_notes returns True for complete notes."""
        valid_md = (
            "# Executive Summary\n"
            "An overview of transformers.\n\n"
            "# Key Concepts\n"
            "- Self-attention mechanism.\n\n"
            "# Action Items\n"
            "- Pretrain on large text corpora."
        )
        self.assertTrue(validate_markdown_notes(valid_md))

    def test_validation_fails_on_missing_executive_summary(self):
        """Test validation fails when Executive Summary header is missing."""
        invalid_md = (
            "# Key Concepts\n"
            "- Concept A\n\n"
            "# Action Items\n"
            "- Action A"
        )
        with self.assertRaises(MarkdownValidationError) as ctx:
            validate_markdown_notes(invalid_md)
        self.assertIn("Executive Summary", str(ctx.exception))

    def test_validation_fails_on_missing_action_items(self):
        """Test validation fails when Action Items header is missing."""
        invalid_md = (
            "# Executive Summary\n"
            "Overview here.\n\n"
            "# Key Concepts\n"
            "- Concept A"
        )
        with self.assertRaises(MarkdownValidationError) as ctx:
            validate_markdown_notes(invalid_md)
        self.assertIn("Action Items", str(ctx.exception))

    def test_validation_fails_on_empty_section_content(self):
        """Test validation fails when a section header exists but has no content."""
        empty_section_md = (
            "# Executive Summary\n"
            "Overview text.\n\n"
            "# Key Concepts\n\n"
            "# Action Items\n"
            "Action text."
        )
        with self.assertRaises(MarkdownValidationError) as ctx:
            validate_markdown_notes(empty_section_md)
        self.assertIn("Key Concepts (empty content)", str(ctx.exception))

    def test_strip_emojis(self):
        """Test removal of decorative emojis while preserving text."""
        raw = "🚀 Introduction to Machine Learning 💡 with Python 🐍"
        cleaned = strip_emojis(raw)
        self.assertEqual(cleaned, "Introduction to Machine Learning  with Python")

    def test_format_action_items_fallback(self):
        """Test formatting when no action items were found."""
        data = {
            "executive_summary": "Lecture on quantum computing history.",
            "key_concepts": ["**Qubits**: Superposition states."],
            "action_items": ["No explicit action items were identified in the transcript."],
        }
        md = format_as_markdown(data)
        self.assertIn("No explicit action items were identified in the transcript.", md)
        self.assertTrue(validate_markdown_notes(md))

    def test_sequential_step_numbering_for_procedures(self):
        """Test that explicit sequential steps format as numbered lists."""
        data = {
            "executive_summary": "Guide on deploying Docker containers.",
            "key_concepts": ["**Containerization**: Operating system level virtualization."],
            "action_items": [
                "Step 1: Write Dockerfile",
                "Step 2: Build container image",
                "Step 3: Run container with port forwarding",
            ],
        }
        md = format_as_markdown(data)
        self.assertIn("1. Step 1: Write Dockerfile", md)
        self.assertIn("2. Step 2: Build container image", md)
        self.assertIn("3. Step 3: Run container with port forwarding", md)

    def test_export_to_docx_package_integrity(self):
        """Test that export_to_docx produces a valid OpenXML Word document (.docx) package."""
        import io
        import zipfile
        from modules.formatter import export_to_docx, _parse_inline_runs

        data = {
            "executive_summary": "Comprehensive analysis of Binary Search Trees and asymptotic complexity.",
            "key_concepts": [
                "**Binary Search Tree**: Left subtrees have smaller keys and right subtrees have larger keys.",
                "**Balanced Trees**: AVL and Red-Black trees guarantee O(log n) worst-case height.",
            ],
            "action_items": [
                "Implement tree insertion and deletion algorithms.",
                "Verify tree balance after node rotations.",
            ],
        }

        docx_bytes = export_to_docx(
            summary_data=data,
            title="Data Structures: Binary Search Trees",
            metadata={"Video ID": "bBC-nXj3Ng4", "Duration": "10m", "Domain": "Computer Science"},
        )

        self.assertIsInstance(docx_bytes, bytes)
        self.assertGreater(len(docx_bytes), 1000)

        # Verify ZIP archive structure
        with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as z:
            names = z.namelist()
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("_rels/.rels", names)
            self.assertIn("word/document.xml", names)
            self.assertIn("word/styles.xml", names)
            self.assertIn("word/_rels/document.xml.rels", names)

            # Check that content XML contains the text and sections
            doc_xml = z.read("word/document.xml").decode("utf-8")
            self.assertIn("Data Structures: Binary Search Trees", doc_xml)
            self.assertIn("Executive Summary", doc_xml)
            self.assertIn("Binary Search Trees and asymptotic complexity", doc_xml)
            self.assertIn("Key Concepts", doc_xml)
            self.assertIn("Binary Search Tree", doc_xml)
            self.assertIn("Action Items", doc_xml)
            self.assertIn("Video ID: bBC-nXj3Ng4", doc_xml)

    def test_parse_inline_runs_formatting(self):
        """Test that _parse_inline_runs breaks down markdown bold and italics."""
        from modules.formatter import _parse_inline_runs

        runs = _parse_inline_runs("Normal text with **bold term** and *italic note*.")
        self.assertEqual(len(runs), 5)
        self.assertEqual(runs[0], ("Normal text with ", False, False))
        self.assertEqual(runs[1], ("bold term", True, False))
        self.assertEqual(runs[2], (" and ", False, False))
        self.assertEqual(runs[3], ("italic note", False, True))
        self.assertEqual(runs[4], (".", False, False))


if __name__ == "__main__":
    unittest.main()


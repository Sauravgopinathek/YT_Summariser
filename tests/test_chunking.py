"""Unit tests for modules.chunking."""

import unittest
from modules.chunking import (
    chunk_transcript,
    get_chunk_statistics,
    split_into_sentences,
)


class TestTranscriptChunking(unittest.TestCase):
    """Test suite for transcript chunking, sentence preservation, and statistics."""

    def test_empty_or_whitespace_input(self):
        """Test that empty or whitespace strings return empty chunk list."""
        self.assertEqual(chunk_transcript(""), [])
        self.assertEqual(chunk_transcript("   \n\t  "), [])
        stats = get_chunk_statistics([])
        self.assertEqual(stats["total_chunks"], 0)
        self.assertEqual(stats["total_words"], 0)

    def test_text_smaller_than_chunk_size(self):
        """Test text smaller than max_words produces exactly one chunk."""
        text = "This is a brief video transcript. It contains only two sentences."
        chunks = chunk_transcript(text, max_words=50, overlap_words=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["word_count"], len(text.split()))

    def test_multi_chunk_splitting_and_sentence_preservation(self):
        """Test splitting text into multiple chunks while keeping sentences intact."""
        sentences = [
            f"Sentence number {i} discusses an interesting topic in detail."
            for i in range(1, 21)
        ]
        # Each sentence has 9 words. 20 sentences = 180 words.
        full_text = " ".join(sentences)

        # Chunks of max 30 words (approx 3 sentences each) without overlap
        chunks = chunk_transcript(full_text, max_words=30, overlap_words=0)

        self.assertGreater(len(chunks), 1)
        # Verify chunks preserve original sequential order
        for idx, chunk in enumerate(chunks, start=1):
            self.assertEqual(chunk["chunk_id"], idx)
            self.assertGreater(chunk["word_count"], 0)
            self.assertLessEqual(chunk["word_count"], 30)

        # Verify no words are silently lost: combined text contains all sentences
        joined_chunks = " ".join(c["text"] for c in chunks)
        for s in sentences:
            self.assertIn(s, joined_chunks)

    def test_overlap_continuity(self):
        """Test that chunks contain expected overlap for context preservation."""
        sentences = [
            "Alpha sentence explains the foundation.",  # 5 words
            "Beta sentence builds on the foundation.",  # 6 words
            "Gamma sentence introduces new concepts.",  # 5 words
            "Delta sentence wraps up the module.",      # 6 words
        ]
        text = " ".join(sentences)
        # max_words=12, overlap_words=6
        chunks = chunk_transcript(text, max_words=12, overlap_words=6)

        self.assertGreater(len(chunks), 1)
        # Check that consecutive chunks share overlapping sentences
        # The trailing sentence of chunk 1 should appear in chunk 2
        chunk1_text = chunks[0]["text"]
        chunk2_text = chunks[1]["text"]
        self.assertIn("Beta sentence builds on the foundation.", chunk1_text)
        self.assertIn("Beta sentence builds on the foundation.", chunk2_text)

    def test_oversized_sentence_handling(self):
        """Test handling of a single long sentence without punctuation exceeding max_words."""
        long_sentence = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_transcript(long_sentence, max_words=25, overlap_words=5)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["word_count"], 25)
            self.assertGreater(chunk["word_count"], 0)

        # Confirm all words from word0 to word99 are covered
        joined = " ".join(c["text"] for c in chunks)
        for i in range(100):
            self.assertIn(f"word{i}", joined)

    def test_invalid_parameters(self):
        """Test rejection of invalid chunk sizes and overlap configurations."""
        with self.assertRaises(ValueError):
            chunk_transcript("Sample text", max_words=0)
        with self.assertRaises(ValueError):
            chunk_transcript("Sample text", max_words=-5)
        with self.assertRaises(ValueError):
            chunk_transcript("Sample text", max_words=100, overlap_words=100)
        with self.assertRaises(ValueError):
            chunk_transcript("Sample text", max_words=100, overlap_words=150)
        with self.assertRaises(ValueError):
            chunk_transcript("Sample text", max_words=100, overlap_words=-1)

    def test_chunk_statistics(self):
        """Test statistical calculations on chunk collections."""
        chunks = [
            {"chunk_id": 1, "text": "one two three", "word_count": 3},
            {"chunk_id": 2, "text": "four five six seven", "word_count": 4},
            {"chunk_id": 3, "text": "eight nine", "word_count": 2},
        ]
        stats = get_chunk_statistics(chunks)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["total_words"], 9)
        self.assertEqual(stats["min_chunk_size"], 2)
        self.assertEqual(stats["max_chunk_size"], 4)
        self.assertEqual(stats["avg_chunk_size"], 3.0)


    def test_split_into_sentences(self):
        """Test splitting text on sentence boundaries while preserving punctuation."""
        text = "First sentence. Second sentence with question? Third sentence! Final statement."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "First sentence.")
        self.assertEqual(sentences[1], "Second sentence with question?")
        self.assertEqual(sentences[2], "Third sentence!")
        self.assertEqual(sentences[3], "Final statement.")

        # Empty / whitespace input
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences("   "), [])


if __name__ == "__main__":
    unittest.main()


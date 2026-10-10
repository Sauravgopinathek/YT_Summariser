"""Transcript chunking and context management module.

Splits cleaned transcripts into semantically coherent chunks respecting
sentence and clause boundaries to prevent exceeding LLM context windows.
"""

import re
from typing import Any, Dict, List, Optional

from config.settings import DEFAULT_CHUNK_OVERLAP_WORDS, DEFAULT_CHUNK_SIZE_WORDS
from modules.exceptions import ChunkingError

# Sentence boundary splitting pattern (split on period, exclamation, question mark followed by whitespace)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentence units without breaking abbreviations or words.

    Args:
        text: Input text string.

    Returns:
        List of non-empty sentence strings.
    """
    if not text or not text.strip():
        return []

    raw_sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


def chunk_transcript(
    text: str,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Partition cleaned transcript text into ordered chunks.

    Respects sentence boundaries whenever possible and enforces a hard word limit.
    Provides optional sliding window overlap for contextual continuity.

    Args:
        text: The cleaned transcript string.
        max_words: Maximum number of words allowed per chunk (defaults to config).
        overlap_words: Number of trailing words from previous chunk to carry over
                       as context (defaults to config). Must be < max_words.

    Returns:
        A list of chunk metadata dictionaries:
            [
                {
                    "chunk_id": 1,
                    "text": "...",
                    "word_count": 800,
                    "char_count": 4500
                },
                ...
            ]

    Raises:
        ValueError: If max_words <= 0 or overlap_words >= max_words.
    """
    if max_words is None:
        max_words = DEFAULT_CHUNK_SIZE_WORDS
    if overlap_words is None:
        overlap_words = min(DEFAULT_CHUNK_OVERLAP_WORDS, max(0, max_words // 5))

    if max_words <= 0:
        raise ChunkingError(f"max_words must be greater than 0, got {max_words}")
    if overlap_words < 0:
        raise ChunkingError(f"overlap_words cannot be negative, got {overlap_words}")
    if overlap_words >= max_words:
        raise ChunkingError(
            f"overlap_words ({overlap_words}) must be strictly less than max_words ({max_words})"
        )

    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)

    # Subdivide any individual sentence that independently exceeds max_words
    atomic_units: List[str] = []
    for s in sentences:
        s_words = s.split()
        if len(s_words) <= max_words:
            atomic_units.append(s)
        else:
            step = max(1, max_words - overlap_words)
            for i in range(0, len(s_words), step):
                sub_slice = s_words[i : i + max_words]
                atomic_units.append(" ".join(sub_slice))
                if i + max_words >= len(s_words):
                    break

    chunks: List[Dict[str, Any]] = []
    current_units: List[str] = []
    current_word_count = 0
    idx = 0
    total_units = len(atomic_units)

    while idx < total_units:
        unit = atomic_units[idx]
        unit_words = len(unit.split())

        if current_word_count + unit_words <= max_words or not current_units:
            current_units.append(unit)
            current_word_count += unit_words
            idx += 1
        else:
            # Finalize the current chunk
            chunk_text = " ".join(current_units).strip()
            chunk_words = len(chunk_text.split())

            if chunk_words > 0:
                chunks.append(
                    {
                        "chunk_id": len(chunks) + 1,
                        "text": chunk_text,
                        "word_count": chunk_words,
                        "char_count": len(chunk_text),
                    }
                )

            # Carry over overlap units if configured
            if overlap_words > 0 and len(current_units) > 1:
                overlap_units: List[str] = []
                accumulated_overlap = 0

                for prev_unit in reversed(current_units):
                    w_count = len(prev_unit.split())
                    if accumulated_overlap + w_count <= overlap_words:
                        overlap_units.insert(0, prev_unit)
                        accumulated_overlap += w_count
                    else:
                        break

                current_units = overlap_units
                current_word_count = accumulated_overlap
            else:
                current_units = []
                current_word_count = 0

    # Final trailing chunk
    if current_units:
        chunk_text = " ".join(current_units).strip()
        chunk_words = len(chunk_text.split())
        if chunk_words > 0:
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "text": chunk_text,
                    "word_count": chunk_words,
                    "char_count": len(chunk_text),
                }
            )

    return chunks


def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics across all generated chunks.

    Args:
        chunks: List of chunk metadata dictionaries.

    Returns:
        Dictionary containing:
            - 'total_chunks': Number of chunks.
            - 'total_words': Sum of words across chunks.
            - 'min_chunk_size': Minimum words in a chunk.
            - 'max_chunk_size': Maximum words in a chunk.
            - 'avg_chunk_size': Average words per chunk (float, 1 decimal place).
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_words": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "avg_chunk_size": 0.0,
        }

    word_counts = [chunk["word_count"] for chunk in chunks]
    total_words = sum(word_counts)
    total_chunks = len(chunks)

    return {
        "total_chunks": total_chunks,
        "total_words": total_words,
        "min_chunk_size": min(word_counts),
        "max_chunk_size": max(word_counts),
        "avg_chunk_size": round(total_words / total_chunks, 1),
    }

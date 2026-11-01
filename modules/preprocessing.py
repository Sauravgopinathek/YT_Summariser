"""Transcript text cleaning and preprocessing module.

Converts raw subtitle segments into coherent, token-efficient text
while strictly preserving technical terminology, formulas, and acronyms.
"""

import html
import re
import unicodedata
from typing import Any, Dict, List

# Known subtitle sound cues and music artifacts
ARTIFACT_PATTERN = re.compile(
    r"\[\s*(music|applause|laughter|cheering|snickers|sighs|cough|groan|groans|inaudible)\s*\]|"
    r"\(\s*(music|applause|laughter|cheering)\s*\)|[♪♫]",
    re.IGNORECASE,
)

# Whitespace normalization pattern
WHITESPACE_PATTERN = re.compile(r"\s+")

# Punctuation spacing fix (e.g., "Hello , world ." -> "Hello, world.")
PUNCT_SPACING_PATTERN = re.compile(r"\s+([,.:;?!])")


def normalize_text(text: str) -> str:
    """Clean and normalize a single text string.

    1. Unescapes HTML entities (e.g., &amp; -> &, &#39; -> ').
    2. Normalizes unicode characters (NFKC) to resolve non-breaking spaces and formatting.
    3. Removes standard subtitle audio cues ([Music], [Applause], ♪, etc.).
    4. Collapses multiple whitespace characters to a single space.

    Args:
        text: Raw text string.

    Returns:
        Cleaned, normalized string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities
    unescaped = html.unescape(text)

    # 2. Unicode normalization (handles \xa0 non-breaking spaces and accents)
    normalized = unicodedata.normalize("NFKC", unescaped)

    # 3. Strip subtitle audio cues and music notes
    no_artifacts = ARTIFACT_PATTERN.sub(" ", normalized)

    # 4. Collapse whitespace
    cleaned = WHITESPACE_PATTERN.sub(" ", no_artifacts).strip()

    return cleaned


def get_transcript_statistics(text: str, segment_count: int = 0) -> Dict[str, Any]:
    """Calculate descriptive statistics for the cleaned transcript.

    Args:
        text: Cleaned transcript text.
        segment_count: Total raw subtitle segments processed.

    Returns:
        Dictionary with clean_text, character_count, word_count, and segment_count.
    """
    clean_text = text if isinstance(text, str) else ""
    words = clean_text.split() if clean_text else []

    return {
        "clean_text": clean_text,
        "character_count": len(clean_text),
        "word_count": len(words),
        "segment_count": max(0, segment_count),
    }


def clean_transcript(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process raw subtitle segments into coherent, contiguous prose.

    Concatenates subtitle segments, resolves fragmented sentences,
    normalizes whitespace and punctuation, while preserving technical
    terminology, acronyms, and formulas.

    Args:
        segments: List of segment dictionaries with at least a 'text' key.

    Returns:
        Dictionary containing:
            - 'clean_text': Coherent processed transcript.
            - 'character_count': Length in characters.
            - 'word_count': Word count of the cleaned text.
            - 'segment_count': Total segments parsed.
    """
    if not segments or not isinstance(segments, list):
        return get_transcript_statistics("", segment_count=0)

    cleaned_parts: List[str] = []

    for seg in segments:
        if isinstance(seg, dict):
            raw_text = seg.get("text", "")
        else:
            raw_text = getattr(seg, "text", "")

        cleaned_seg = normalize_text(str(raw_text))
        if cleaned_seg:
            cleaned_parts.append(cleaned_seg)

    if not cleaned_parts:
        return get_transcript_statistics("", segment_count=len(segments))

    # Stitch segments into continuous text with proper spacing
    stitched_text = " ".join(cleaned_parts)

    # Clean up spaces preceding standard punctuation introduced by segment joins
    stitched_text = PUNCT_SPACING_PATTERN.sub(r"\1", stitched_text)

    # Final whitespace pass
    stitched_text = WHITESPACE_PATTERN.sub(" ", stitched_text).strip()

    return get_transcript_statistics(stitched_text, segment_count=len(segments))

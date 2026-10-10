"""GenAI Summarization Engine module.

Implements grounded technical summarization with direct and hierarchical
(map-reduce) execution strategies and robust Markdown parsing.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from config.settings import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    MAX_TRANSCRIPT_WORDS,
)
from modules.chunking import chunk_transcript
from modules.exceptions import (
    EmptyTranscriptError,
    TranscriptTooLongError,
)

from modules.llm import NVIDIAClient
from prompts.loader import (
    get_chunk_map_prompt,
    get_reduce_prompt,
    get_summary_user_prompt,
    get_system_prompt,
)

logger = logging.getLogger(__name__)


def parse_markdown_notes(markdown_text: str) -> Dict[str, Any]:
    """Parse generated Markdown into executive summary, key concepts, and action items.

    Uses header matching to extract content under each required heading.
    If parsing encounters unconventional formatting, it extracts paragraphs gracefully
    rather than discarding data.

    Args:
        markdown_text: Raw Markdown output from LLM.

    Returns:
        Dictionary with:
            - 'executive_summary': str
            - 'key_concepts': List[str]
            - 'action_items': List[str]
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return {
            "executive_summary": "",
            "key_concepts": [],
            "action_items": [],
        }

    # Normalize newlines
    normalized_text = markdown_text.replace("\r\n", "\n").strip()

    # Split on headers: matches #, ##, or ### followed by section names
    header_pattern = re.compile(
        r"(?m)^#{1,3}\s+(Executive\s+Summary|Key\s+Concepts|Action\s+Items)\b",
        re.IGNORECASE,
    )

    matches = list(header_pattern.finditer(normalized_text))
    sections: Dict[str, str] = {}

    if not matches:
        # Fallback if model omitted standard markdown headers
        logger.warning("No standard headers detected in model output; using raw text fallback.")
        return {
            "executive_summary": normalized_text,
            "key_concepts": [],
            "action_items": [],
        }

    for i, match in enumerate(matches):
        raw_title = match.group(1).lower()
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start_idx:end_idx].strip()

        if "executive" in raw_title:
            sections["executive_summary"] = content
        elif "concept" in raw_title:
            sections["key_concepts"] = content
        elif "action" in raw_title:
            sections["action_items"] = content

    # Helper to parse list items or bullets from a markdown section
    def extract_bullets_or_paragraphs(text_block: str) -> List[str]:
        if not text_block:
            return []
        items = []
        # Find lines starting with bullets (*, -, +, 1., 2., etc.)
        bullet_pattern = re.compile(r"(?m)^\s*(?:[-*+]|\d+\.)\s+(.+)$")
        bullets = bullet_pattern.findall(text_block)

        if bullets:
            for b in bullets:
                cleaned = b.strip()
                if cleaned:
                    items.append(cleaned)
        else:
            # Fall back to splitting on double newlines (paragraphs)
            paragraphs = [p.strip() for p in text_block.split("\n\n") if p.strip()]
            items = paragraphs

        return items

    exec_summary = sections.get("executive_summary", "")
    key_concepts_raw = sections.get("key_concepts", "")
    action_items_raw = sections.get("action_items", "")

    key_concepts_list = extract_bullets_or_paragraphs(key_concepts_raw)
    action_items_list = extract_bullets_or_paragraphs(action_items_raw)

    return {
        "executive_summary": exec_summary,
        "key_concepts": key_concepts_list,
        "action_items": action_items_list,
    }


def generate_notes(
    transcript: str,
    client: Optional[NVIDIAClient] = None,
    max_words_per_chunk: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate structured reference notes from a cleaned video transcript.

    Chooses between:
    - Direct generation (for transcripts fitting into a single chunk)
    - Hierarchical Map-Reduce (for long multi-chunk transcripts)

    Args:
        transcript: Cleaned, normalized transcript text.
        client: Optional NVIDIAClient instance.
        max_words_per_chunk: Maximum words per chunk.
        overlap_words: Words of overlap between consecutive chunks.

    Returns:
        Structured dictionary containing:
            - 'executive_summary': str
            - 'key_concepts': List[str]
            - 'action_items': List[str]
            - 'markdown': str (full Markdown output)
            - 'metadata': dict with performance and processing metrics

    Raises:
        ValueError: If transcript is empty.
        LLMError: If LLM inference fails.
    """
    if not transcript or not transcript.strip():
        raise EmptyTranscriptError("Cannot generate notes: Transcript text is empty.")

    clean_text = transcript.strip()
    words = clean_text.split()
    total_words = len(words)
    total_chars = len(clean_text)

    if total_words > MAX_TRANSCRIPT_WORDS:
        raise TranscriptTooLongError(
            f"Transcript word count ({total_words:,}) exceeds maximum allowed limit "
            f"({MAX_TRANSCRIPT_WORDS:,} words). Please process a shorter transcript."
        )

    llm_client = client or NVIDIAClient()
    system_prompt = get_system_prompt()

    chunk_size = max_words_per_chunk if max_words_per_chunk is not None else DEFAULT_CHUNK_SIZE_WORDS
    if overlap_words is not None:
        overlap = overlap_words
    else:
        overlap = min(DEFAULT_CHUNK_OVERLAP_WORDS, max(0, chunk_size // 5))

    chunks = chunk_transcript(clean_text, max_words=chunk_size, overlap_words=overlap)
    chunk_count = max(1, len(chunks))

    start_time = time.perf_counter()
    llm_call_count = 0
    final_markdown = ""
    strategy = "direct"

    # Strategy Selection: Direct vs. Map-Reduce
    if len(chunks) <= 1:
        strategy = "direct"
        logger.info("Executing Direct Summarization strategy (words=%d)", total_words)
        user_prompt = get_summary_user_prompt(clean_text)

        final_markdown = llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        llm_call_count += 1
    else:
        strategy = "map_reduce"
        logger.info(
            "Executing Map-Reduce Summarization strategy across %d chunks (total_words=%d)",
            len(chunks),
            total_words,
        )

        # Step 1: Map - summarize each chunk
        chunk_summaries: List[str] = []
        for chunk in chunks:
            c_id = chunk["chunk_id"]
            map_prompt = get_chunk_map_prompt(
                chunk_text=chunk["text"],
                chunk_id=c_id,
                total_chunks=len(chunks),
            )
            logger.info("Map step: processing chunk %d/%d", c_id, len(chunks))
            summary_part = llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=map_prompt,
            )
            chunk_summaries.append(summary_part)
            llm_call_count += 1

        # Step 2: Combine chunk summaries
        combined_summaries_text = "\n\n".join(
            [
                f"### Section {c['chunk_id']} Summary ({c['word_count']} words):\n{s}"
                for c, s in zip(chunks, chunk_summaries)
            ]
        )

        # Step 3: Reduce - synthesize final structured notes
        logger.info("Reduce step: synthesizing final structured summary")
        reduce_prompt = get_reduce_prompt(combined_summaries_text)
        final_markdown = llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=reduce_prompt,
        )
        llm_call_count += 1

    elapsed_time = round(time.perf_counter() - start_time, 2)

    # Parse structured Markdown sections
    parsed_sections = parse_markdown_notes(final_markdown)

    result = {
        "executive_summary": parsed_sections["executive_summary"],
        "key_concepts": parsed_sections["key_concepts"],
        "action_items": parsed_sections["action_items"],
        "markdown": final_markdown,
        "metadata": {
            "chunk_count": len(chunks),
            "llm_call_count": llm_call_count,
            "input_characters": total_chars,
            "input_words": total_words,
            "output_characters": len(final_markdown),
            "output_words": len(final_markdown.split()),
            "processing_time_seconds": elapsed_time,
            "strategy": strategy,
        },
    }

    logger.info(
        "Summarization complete (strategy=%s, calls=%d, duration=%.2fs)",
        strategy,
        llm_call_count,
        elapsed_time,
    )
    return result


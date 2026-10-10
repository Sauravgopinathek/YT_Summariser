"""Evaluation and performance metrics module.

Provides precise latency tracking, compression ratio calculation, and formal
ROUGE (1/2/L) scoring for evaluating video summarization quality against
genuine ground-truth reference summaries.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PerformanceMetrics:
    """Detailed performance and latency telemetry for an end-to-end run."""

    url_parsing_latency_seconds: float = 0.0
    transcript_retrieval_latency_seconds: float = 0.0
    preprocessing_latency_seconds: float = 0.0
    chunking_latency_seconds: float = 0.0
    llm_inference_latency_seconds: float = 0.0
    total_processing_time_seconds: float = 0.0

    input_word_count: int = 0
    output_word_count: int = 0
    compression_ratio: float = 0.0  # output_words / input_words
    reduction_percentage: float = 0.0  # (1 - compression_ratio) * 100

    chunk_count: int = 0
    llm_call_count: int = 0

    tokens_per_second: Optional[float] = None
    time_to_first_token_seconds: Optional[float] = None
    faithfulness_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return asdict(self)


def calculate_compression_ratio(transcript_words: int, summary_words: int) -> Tuple[float, float]:
    """Calculate the compression ratio and percentage reduction.

    Args:
        transcript_words: Total words in the clean source transcript.
        summary_words: Total words in the generated summary.

    Returns:
        Tuple of (compression_ratio, reduction_percentage).
            - compression_ratio: summary_words / transcript_words (e.g., 0.15 = summary is 15% size of transcript).
            - reduction_percentage: percentage reduction in content volume (e.g., 85.0%).
    """
    if transcript_words <= 0:
        return 0.0, 0.0

    ratio = round(summary_words / transcript_words, 4)
    reduction = round(max(0.0, (1.0 - ratio) * 100.0), 2)
    return ratio, reduction


def _tokenize(text: str) -> List[str]:
    """Basic alphanumeric word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Generate n-grams from a token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute the Longest Common Subsequence (LCS) length between two sequences."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def calculate_rouge(
    candidate_summary: str,
    reference_summary: str,
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores against a genuine reference summary.

    Note: ROUGE scores should strictly be calculated when a genuine ground-truth reference
    summary exists. Never fabricate or hallucinate reference summaries.

    Args:
        candidate_summary: Generated notes or summary string.
        reference_summary: Ground-truth reference summary string.

    Returns:
        Dictionary mapping 'rouge-1', 'rouge-2', 'rouge-l' to precision, recall, and f1.

    Raises:
        ValueError: If either summary is empty or whitespace.
    """
    if not candidate_summary or not candidate_summary.strip():
        raise ValueError("Cannot calculate ROUGE: candidate_summary is empty.")
    if not reference_summary or not reference_summary.strip():
        raise ValueError("Cannot calculate ROUGE: reference_summary is empty.")

    cand_tokens = _tokenize(candidate_summary)
    ref_tokens = _tokenize(reference_summary)

    if not cand_tokens or not ref_tokens:
        raise ValueError("Tokenization yielded no tokens for ROUGE evaluation.")

    scores: Dict[str, Dict[str, float]] = {}

    # Helper for N-gram scores (ROUGE-1, ROUGE-2)
    for n, metric_name in [(1, "rouge-1"), (2, "rouge-2")]:
        cand_ngrams = _get_ngrams(cand_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not cand_ngrams or not ref_ngrams:
            scores[metric_name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue

        cand_counter = Counter(cand_ngrams)
        ref_counter = Counter(ref_ngrams)

        overlap = sum((cand_counter & ref_counter).values())
        precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        scores[metric_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # ROUGE-L: Longest Common Subsequence
    lcs = _lcs_length(cand_tokens, ref_tokens)
    p_l = lcs / len(cand_tokens) if cand_tokens else 0.0
    r_l = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1_l = (2 * p_l * r_l) / (p_l + r_l) if (p_l + r_l) > 0 else 0.0

    scores["rouge-l"] = {
        "precision": round(p_l, 4),
        "recall": round(r_l, 4),
        "f1": round(f1_l, 4),
    }

    return scores

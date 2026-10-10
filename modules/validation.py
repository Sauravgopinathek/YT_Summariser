"""Summary grounding, faithfulness, and hallucination detection module.

Provides a multi-layered heuristic validation engine to detect unsupported claims,
missing terminology, and hallucinated numerical figures, with an optional
LLM-based verification pass.

DISCLAIMER: Automated heuristic validation is an approximate metric to assist review
and does NOT guarantee that text is 100% free of subtle hallucinations.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union

from modules.formatter import validate_markdown_notes
from modules.llm import NVIDIAClient

logger = logging.getLogger(__name__)

# Pattern to capture numerical figures, percentages, dates, and currency
NUMERICAL_CLAIM_PATTERN = re.compile(
    r"(?:\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d{4}\b|\b\d+(?:,\d{3})+(?:\.\d+)?|\b\d+\.\d+\b)"
)

# Pattern to extract acronyms and technical identifiers (e.g. GPU, PyTorch, CNN, LLM, API)
TECHNICAL_TERM_PATTERN = re.compile(
    r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b|\b[A-Z][a-zA-Z0-9_]{3,}\b"
)

# Common words matching uppercase pattern that are not technical terms
STOP_ACRONYMS = {"THE", "AND", "FOR", "THIS", "THAT", "WITH", "FROM", "WHAT", "WHEN", "WHERE", "YOUR"}


def extract_numerical_claims(text: str) -> Set[str]:
    """Extract specific numbers, percentages, currency, and dates from text.

    Args:
        text: Source text string.

    Returns:
        Set of standardized numerical strings.
    """
    if not text:
        return set()
    raw_matches = NUMERICAL_CLAIM_PATTERN.findall(text)
    # Normalize matches (strip whitespace, unify formatting)
    normalized = {re.sub(r"\s+", "", m).strip() for m in raw_matches if m.strip()}
    return normalized


def extract_key_terminology(text: str) -> Set[str]:
    """Extract prominent technical acronyms, frameworks, and identifiers.

    Args:
        text: Source text string.

    Returns:
        Set of prominent technical terms found in the text.
    """
    if not text:
        return set()
    raw_terms = TECHNICAL_TERM_PATTERN.findall(text)
    terms = {t.strip() for t in raw_terms if t.strip().upper() not in STOP_ACRONYMS and len(t) >= 2}
    return terms


def validate_summary_faithfulness(
    transcript: str,
    summary_data: Union[Dict[str, Any], str],
    min_acceptable_score: float = 0.6,
) -> Dict[str, Any]:
    """Evaluate grounding and faithfulness of generated notes against the source transcript.

    Heuristic checks performed:
    1. Structural completeness (Executive Summary, Key Concepts, Action Items exist).
    2. Non-empty content verification.
    3. Technical terminology preservation (acronyms and framework names).
    4. Unsupported numerical figures (numbers appearing in notes absent in transcript).
    5. Faithfulness score calculation.

    Args:
        transcript: Cleaned original transcript text.
        summary_data: Dictionary containing summary fields or raw Markdown text.
        min_acceptable_score: Threshold below which valid=False.

    Returns:
        Dictionary containing:
            - 'valid': bool
            - 'faithfulness_score': float (0.0 to 1.0)
            - 'coverage_score': float (0.0 to 1.0)
            - 'unsupported_numbers': List[str]
            - 'preserved_terms': List[str]
            - 'missing_terms': List[str]
            - 'warnings': List[str]
            - 'disclaimer': str
    """
    warnings: List[str] = []

    # 1. Normalize summary text
    if isinstance(summary_data, dict):
        full_text = summary_data.get("markdown") or ""
        if not full_text:
            exec_s = summary_data.get("executive_summary", "")
            kc = " ".join(summary_data.get("key_concepts", []))
            ai = " ".join(summary_data.get("action_items", []))
            full_text = f"{exec_s}\n{kc}\n{ai}".strip()
    else:
        full_text = str(summary_data).strip()

    # 2. Basic Non-Empty Check
    if not full_text:
        return {
            "valid": False,
            "faithfulness_score": 0.0,
            "coverage_score": 0.0,
            "unsupported_numbers": [],
            "preserved_terms": [],
            "missing_terms": [],
            "warnings": ["Generated summary is completely empty."],
            "disclaimer": "Automated heuristic metric. Does not guarantee 100% hallucination-free output.",
        }

    # 3. Structure Check
    try:
        validate_markdown_notes(full_text)
    except Exception as e:
        warnings.append(f"Structure validation issue: {e}")

    # 4. Numerical Claims Verification (Detect Hallucinated Numbers)
    transcript_numbers = extract_numerical_claims(transcript)
    summary_numbers = extract_numerical_claims(full_text)

    # Numbers in summary not present in the transcript (ignoring small list numbering 1-9)
    unsupported_numbers: List[str] = []
    for num in summary_numbers:
        # Check direct or stripped match
        num_clean = num.replace("$", "").replace("%", "").replace(",", "")
        transcript_numbers_clean = {t.replace("$", "").replace("%", "").replace(",", "") for t in transcript_numbers}

        if num not in transcript_numbers and num_clean not in transcript_numbers_clean:
            # Skip single-digit integers which might be list indicators
            if not (num_clean.isdigit() and len(num_clean) == 1):
                unsupported_numbers.append(num)

    if unsupported_numbers:
        warnings.append(
            f"Detected {len(unsupported_numbers)} specific numerical claim(s) in the summary that do not appear "
            f"in the transcript: {', '.join(sorted(unsupported_numbers)[:5])}. Please review these claims manually."
        )

    # 5. Technical Terminology Preservation
    transcript_terms = extract_key_terminology(transcript)
    summary_terms = extract_key_terminology(full_text)

    preserved_terms: List[str] = []
    missing_terms: List[str] = []

    for term in sorted(transcript_terms):
        if term.lower() in full_text.lower():
            preserved_terms.append(term)
        else:
            missing_terms.append(term)

    coverage_score = (
        round(len(preserved_terms) / len(transcript_terms), 2)
        if transcript_terms
        else 1.0
    )

    # 6. Compute Faithfulness Score (Heuristic)
    # Base score is 1.0; penalized by unsupported numbers and low term coverage
    score = 1.0

    # Penalize 0.15 per unsupported number (up to 0.45 max penalty)
    num_penalty = min(0.45, len(unsupported_numbers) * 0.15)
    score -= num_penalty

    # Weight term preservation into faithfulness
    if transcript_terms:
        score = round((score * 0.7) + (coverage_score * 0.3), 2)
    else:
        score = round(max(0.0, score), 2)

    is_valid = score >= min_acceptable_score and len(warnings) < 3

    if score < min_acceptable_score:
        warnings.append(
            f"Faithfulness score ({score:.2f}) is below acceptable threshold ({min_acceptable_score:.2f})."
        )

    logger.info(
        "Validation complete (faithfulness=%.2f, coverage=%.2f, unsupported_nums=%d, warnings=%d)",
        score,
        coverage_score,
        len(unsupported_numbers),
        len(warnings),
    )

    return {
        "valid": is_valid,
        "faithfulness_score": score,
        "coverage_score": coverage_score,
        "unsupported_numbers": sorted(unsupported_numbers),
        "preserved_terms": preserved_terms[:15],
        "missing_terms": missing_terms[:10],
        "warnings": warnings,
        "disclaimer": (
            "Heuristic grounding metric: assesses term preservation and unsupported numbers. "
            "Not an absolute guarantee against subtle hallucination."
        ),
    }


def verify_faithfulness_with_llm(
    transcript: str,
    summary_markdown: str,
    client: Optional[NVIDIAClient] = None,
) -> Dict[str, Any]:
    """Perform an optional second-pass LLM verification of factual grounding.

    Args:
        transcript: Full source transcript text.
        summary_markdown: Formatted markdown notes to audit.
        client: Optional NVIDIAClient instance.

    Returns:
        Dictionary containing:
            - 'llm_verified': bool
            - 'unsupported_claims': List[str]
            - 'verdict': str
            - 'analysis': str
    """
    llm_client = client or NVIDIAClient()

    system_prompt = (
        "You are an impartial fact-checking and grounding auditor. "
        "Your sole task is to verify whether claims in a summary are directly supported by the source transcript. "
        "Strictly identify any hallucinations, invented examples, or unsupported facts."
    )

    user_prompt = f"""Source Transcript:
\"\"\"
{transcript[:8000]}
\"\"\"

Summary Notes to Verify:
\"\"\"
{summary_markdown}
\"\"\"

Instructions:
1. Identify any statements in the summary that cannot be directly supported by the transcript.
2. If all claims are supported, state: "VERDICT: FAITHFUL".
3. If unsupported claims exist, list each unsupported statement and state: "VERDICT: UNSUPPORTED CLAIMS FOUND".
Keep your response concise.
"""

    try:
        raw_response = llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=600,
        )

        is_faithful = "VERDICT: FAITHFUL" in raw_response.upper()
        return {
            "llm_verified": True,
            "is_faithful": is_faithful,
            "analysis": raw_response.strip(),
        }
    except Exception as e:
        logger.warning("LLM verification pass failed: %s", e)
        return {
            "llm_verified": False,
            "is_faithful": None,
            "analysis": f"LLM verification could not be completed: {e}",
        }

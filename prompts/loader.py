"""Prompt loader module for reading externalized prompt templates."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"
SUMMARY_PROMPT_FILE = PROMPTS_DIR / "summary_prompt.txt"
CHUNK_SUMMARY_PROMPT_FILE = PROMPTS_DIR / "chunk_summary_prompt.txt"
REDUCE_SUMMARY_PROMPT_FILE = PROMPTS_DIR / "reduce_summary_prompt.txt"


def load_prompt_file(file_path: Path) -> str:
    """Read and return content of a prompt template file.

    Args:
        file_path: Path object to the prompt text file.

    Returns:
        Clean stripped string content of the prompt file.

    Raises:
        FileNotFoundError: If the prompt template cannot be located.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template file not found at: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def get_system_prompt() -> str:
    """Retrieve the core system prompt for transcript summarization."""
    return load_prompt_file(SYSTEM_PROMPT_FILE)


def get_summary_user_prompt(transcript: str) -> str:
    """Format the direct user summary prompt with the supplied transcript text."""
    template = load_prompt_file(SUMMARY_PROMPT_FILE)
    clean_text = transcript.strip() if transcript else ""
    return template.replace("{transcript}", clean_text)


def get_chunk_map_prompt(chunk_text: str, chunk_id: int, total_chunks: int) -> str:
    """Format the chunk-level map prompt for intermediate chunk summarization."""
    template = load_prompt_file(CHUNK_SUMMARY_PROMPT_FILE)
    return (
        template.replace("{chunk_id}", str(chunk_id))
        .replace("{total_chunks}", str(total_chunks))
        .replace("{chunk_text}", chunk_text.strip() if chunk_text else "")
    )


def get_reduce_prompt(combined_summaries: str) -> str:
    """Format the reduce prompt for synthesizing all chunk summaries."""
    template = load_prompt_file(REDUCE_SUMMARY_PROMPT_FILE)
    return template.replace("{combined_summaries}", combined_summaries.strip() if combined_summaries else "")

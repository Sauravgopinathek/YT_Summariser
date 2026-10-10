"""Prompts package containing system and structured prompt templates."""

from prompts.loader import (
    get_summary_user_prompt,
    get_system_prompt,
    load_prompt_file,
)

__all__ = [
    "get_system_prompt",
    "get_summary_user_prompt",
    "load_prompt_file",
]

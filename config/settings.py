"""Application settings and configuration parameters."""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv(override=True)

# Chunking Configuration
DEFAULT_CHUNK_SIZE_WORDS = int(os.getenv("DEFAULT_CHUNK_SIZE_WORDS", "800"))
DEFAULT_CHUNK_OVERLAP_WORDS = int(os.getenv("DEFAULT_CHUNK_OVERLAP_WORDS", "100"))
MAX_TRANSCRIPT_WORDS = int(os.getenv("MAX_TRANSCRIPT_WORDS", "50000"))

# NVIDIA NIM / LLM Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))

# UI Settings
PAGE_TITLE = "YouTube Video Summarizer & Notes Extractor"
PAGE_ICON = "🎬"


def get_nvidia_api_key() -> str:
    """Retrieve the latest NVIDIA API key from environment, re-reading .env if needed."""
    load_dotenv(override=True)
    return os.getenv("NVIDIA_API_KEY", "").strip()

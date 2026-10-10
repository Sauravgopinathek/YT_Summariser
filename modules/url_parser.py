"""YouTube URL parser and Video ID extractor module.

This module provides utilities to validate YouTube URLs and extract
the standard 11-character video ID without any network requests.
"""

import re
from urllib.parse import parse_qs, urlparse

from modules.exceptions import InvalidYouTubeURLError


# YouTube video IDs are strictly 11 characters consisting of base64url characters: [a-zA-Z0-9_-]
VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Recognized YouTube domain hostnames
VALID_HOSTNAMES = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_video_id(url: str) -> str:
    """Extract and validate the 11-character video ID from a YouTube URL.

    Supported formats include:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - URLs with extra query params (?si=..., &t=..., &feature=...)

    Args:
        url: The YouTube URL string to parse.

    Returns:
        The extracted 11-character YouTube video ID string.

    Raises:
        InvalidYouTubeURLError: If the URL is empty, malformed, not a recognized
            YouTube domain, or the video ID is missing / not exactly 11 characters.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise InvalidYouTubeURLError("YouTube URL cannot be empty.")

    cleaned_url = url.strip()

    # Prepend scheme if omitted to allow urlparse to correctly identify netloc
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    try:
        parsed = urlparse(cleaned_url)
    except Exception as e:
        raise InvalidYouTubeURLError(f"Malformed URL provided: {e}") from e

    hostname = (parsed.netloc or "").lower().split(":")[0]  # remove port if present

    if hostname not in VALID_HOSTNAMES:
        raise InvalidYouTubeURLError(
            f"Unrecognized YouTube domain '{parsed.netloc}'. Expected one of: {', '.join(sorted(VALID_HOSTNAMES))}"
        )

    candidate_id = ""

    if hostname in {"youtu.be", "www.youtu.be"}:
        # Format: https://youtu.be/<VIDEO_ID>
        path_segments = [seg for seg in parsed.path.split("/") if seg]
        if path_segments:
            candidate_id = path_segments[0]
        else:
            raise InvalidYouTubeURLError("Missing video ID in shortened youtu.be URL.")
    else:
        # Standard youtube.com domain formats
        path = parsed.path

        if path.startswith("/watch"):
            query_params = parse_qs(parsed.query)
            v_params = query_params.get("v")
            if not v_params or not v_params[0]:
                raise InvalidYouTubeURLError("Missing 'v' parameter in YouTube watch URL.")
            candidate_id = v_params[0]

        elif path.startswith(("/embed/", "/v/", "/shorts/")):
            path_segments = [seg for seg in path.split("/") if seg]
            # e.g., ['embed', '<VIDEO_ID>'] or ['v', '<VIDEO_ID>'] or ['shorts', '<VIDEO_ID>']
            if len(path_segments) >= 2:
                candidate_id = path_segments[1]
            else:
                raise InvalidYouTubeURLError(f"Missing video ID in '{path}' URL path.")
        else:
            raise InvalidYouTubeURLError(
                f"Unsupported YouTube URL path: '{path}'. Expected '/watch', '/embed/', '/shorts/', or 'youtu.be/<ID>'."
            )

    # Validate candidate ID
    if len(candidate_id) != 11:
        raise InvalidYouTubeURLError(
            f"Invalid video ID length ({len(candidate_id)} characters). Expected exactly 11 characters. Found: '{candidate_id}'"
        )

    if not VIDEO_ID_REGEX.match(candidate_id):
        raise InvalidYouTubeURLError(
            f"Video ID '{candidate_id}' contains invalid characters. Must contain only letters, digits, '-', or '_'."
        )

    return candidate_id


def validate_youtube_url(url: str) -> bool:
    """Check if the provided string is a valid YouTube URL containing an 11-char video ID.

    Args:
        url: The URL string to test.

    Returns:
        True if valid, False otherwise.
    """
    try:
        extract_video_id(url)
        return True
    except (InvalidYouTubeURLError, TypeError, ValueError):
        return False


def validate_video_id(video_id: str) -> bool:
    """Check if the provided string is a valid 11-character YouTube video ID.

    Args:
        video_id: Video ID string to test.

    Returns:
        True if exactly 11 valid base64url characters, False otherwise.
    """
    return bool(
        video_id
        and isinstance(video_id, str)
        and VIDEO_ID_REGEX.match(video_id.strip())
    )


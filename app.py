"""YouTube Video Summarizer & Notes Extractor Streamlit Dashboard.

Zero-download Generative AI application that extracts YouTube video transcripts
and uses NVIDIA NIM LLM endpoints to generate structured reference notes with
production-grade error handling, secret sanitization, context management,
and automated grounding/faithfulness validation.
"""

import streamlit as st

from config.settings import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    PAGE_ICON,
    PAGE_TITLE,
    get_nvidia_api_key,
)
from modules.exceptions import (
    CaptionsDisabledError,
    EnglishTranscriptNotFoundError,
)
from modules.formatter import (
    export_to_docx,
    format_as_markdown,
    validate_markdown_notes,
)

from modules.llm import NVIDIAClient
from modules.preprocessing import clean_transcript
from modules.summarizer import generate_notes
from modules.transcript import YouTubeTranscriptProvider
from modules.url_parser import extract_video_id
from modules.validation import (
    validate_summary_faithfulness,
    verify_faithfulness_with_llm,
)
from utils.error_handler import format_user_error


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_cached_transcript(video_id: str):
    """Retrieve and cache raw transcript segments for a given video ID."""
    provider = YouTubeTranscriptProvider()
    return provider.get_transcript(video_id)


# Page Setup
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

# Hide Streamlit Cloud header toolbar and bottom GitHub avatar footer badge
st.markdown(
    """
    <style>
    /* 1. Hide top header toolbar and decoration */
    header [data-testid="stHeaderActionElements"],
    header [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
    }

    /* 2. Hide footer and all known badge selectors */
    footer,
    [data-testid="stFooter"],
    [data-testid="manage-app-button"],
    div[class*="viewerBadge"],
    a[class*="viewerBadge"],
    span[class*="viewerBadge"],
    div[class*="manageApp"],
    div[class*="StatusWidget"],
    div[class*="stAppDeployButton"],
    div[class*="DeployButton"],
    div[class*="stBottom"],
    div[class*="ProfileBadge"],
    div[class*="profileBadge"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
    }

    /* 3. Nuclear: hide any direct child of body that is NOT #root */
    body > :not(#root):not(script):not(style):not(link):not(noscript):not(meta) {
        display: none !important;
        visibility: hidden !important;
    }

    /* 4. Hide any floating fixed/absolute pill at bottom-right corners */
    div[style*="position: fixed"][style*="bottom"],
    div[style*="position: fixed"][style*="right"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Client-side JavaScript to actively find and destroy the badge by DOM walking
import streamlit.components.v1 as components
components.html(
    """
    <script>
    (function() {
        function nukeBySelectors(doc) {
            const sels = [
                'footer', '[data-testid="stFooter"]',
                '[data-testid="stStatusWidget"]',
                '[data-testid="manage-app-button"]',
                'div[class*="viewerBadge"]', 'a[class*="viewerBadge"]',
                'span[class*="viewerBadge"]', 'div[class*="manageApp"]',
                'div[class*="StatusWidget"]', 'div[class*="ProfileBadge"]',
                'div[class*="profileBadge"]', 'div[class*="stAppDeployButton"]',
                'div[class*="DeployButton"]', 'div[class*="stBottom"]'
            ];
            sels.forEach(s => doc.querySelectorAll(s).forEach(el => el.remove()));
        }

        function nukeByStructure(doc) {
            // Remove any body child that isn't #root, script, style, link
            if (doc.body) {
                Array.from(doc.body.children).forEach(child => {
                    const tag = child.tagName.toLowerCase();
                    if (child.id === 'root') return;
                    if (['script','style','link','noscript','meta','iframe'].includes(tag)) return;
                    // If it's a small fixed/absolute positioned element, it's likely the badge
                    const cs = doc.defaultView ? doc.defaultView.getComputedStyle(child) : null;
                    if (cs && (cs.position === 'fixed' || cs.position === 'absolute')) {
                        child.remove();
                        return;
                    }
                    // If it contains an img with border-radius (avatar circle), remove it
                    const imgs = child.querySelectorAll('img');
                    if (imgs.length > 0 && child.offsetHeight < 80 && child.offsetWidth < 80) {
                        child.remove();
                        return;
                    }
                });
            }

            // Also check inside #root for siblings of .stApp
            const root = doc.getElementById('root');
            if (root) {
                Array.from(root.children).forEach(child => {
                    if (child.classList && child.classList.contains('stApp')) return;
                    const tag = child.tagName.toLowerCase();
                    if (['script','style','link','noscript'].includes(tag)) return;
                    // Anything else outside .stApp is likely injected badge
                    const cs = doc.defaultView ? doc.defaultView.getComputedStyle(child) : null;
                    if (cs && (cs.position === 'fixed' || cs.position === 'absolute')) {
                        child.remove();
                    }
                });
            }
        }

        function injectCSS(doc) {
            if (doc.getElementById('__nuke_badge')) return;
            const s = doc.createElement('style');
            s.id = '__nuke_badge';
            s.textContent = `
                body > :not(#root):not(script):not(style):not(link):not(noscript):not(meta) {
                    display: none !important;
                    visibility: hidden !important;
                }
                footer, [data-testid="stFooter"],
                div[class*="viewerBadge"], div[class*="manageApp"],
                div[class*="StatusWidget"], div[class*="ProfileBadge"],
                div[class*="profileBadge"], div[class*="stAppDeployButton"],
                div[class*="DeployButton"], div[class*="stBottom"],
                [data-testid="manage-app-button"] {
                    display: none !important;
                    visibility: hidden !important;
                    height: 0 !important; width: 0 !important;
                }
            `;
            (doc.head || doc.documentElement).appendChild(s);
        }

        function fullPurge(doc) {
            injectCSS(doc);
            nukeBySelectors(doc);
            nukeByStructure(doc);
        }

        function setup(doc) {
            fullPurge(doc);
            const obs = new MutationObserver(() => fullPurge(doc));
            obs.observe(doc.body || doc.documentElement, { childList: true, subtree: true });
        }

        // Run on all reachable document levels
        try { setup(document); } catch(e) {}
        try {
            const pd = window.parent.document;
            if (pd && pd !== document) setup(pd);
        } catch(e) {}
        try {
            const td = window.top.document;
            if (td && td !== document && td !== window.parent.document) setup(td);
        } catch(e) {}
    })();
    </script>
    """,
    height=0,
    width=0,
)


st.title("YouTube Video Summarizer & Notes Extractor")

st.markdown("**Zero-download GenAI summarization of YouTube educational content.**")

st.markdown("---")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("NVIDIA NIM / LLM Status")
    current_key = get_nvidia_api_key()
    if current_key:
        masked_key = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "***"
        st.success(f"API Key Detected: `{masked_key}`")
    else:
        st.warning("⚠️ No API Key found in `.env`. Please add `NVIDIA_API_KEY`.")

    st.text(f"Model: {NVIDIA_MODEL}")
    st.caption(f"Endpoint: {NVIDIA_BASE_URL}")

    if st.button("Test NIM Connection", help="Sends a handshake test prompt to verify credentials."):
        with st.spinner("Connecting to NVIDIA NIM..."):
            try:
                test_client = NVIDIAClient()
                reply = test_client.test_connection()
                st.success(f"✅ Handshake Verified!\n\n**Response:** `{reply}`")
            except Exception as err:
                err_title, err_desc = format_user_error(err)
                st.error(f"**{err_title}**: {err_desc}")

    st.markdown("---")
    st.subheader("Chunking & Context Limits")
    chunk_size = st.number_input(
        "Max Words per Chunk:",
        min_value=100,
        max_value=5000,
        value=DEFAULT_CHUNK_SIZE_WORDS,
        step=50,
        help="Target maximum word count per chunk. 800 words is approx. 1,000–1,100 tokens.",
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap (Words):",
        min_value=0,
        max_value=int(chunk_size) - 1,
        value=min(DEFAULT_CHUNK_OVERLAP_WORDS, max(0, int(chunk_size) // 5)),
        step=10,
        help="Number of trailing words from previous chunk for context continuity.",
    )

    st.markdown("---")
    st.subheader("Faithfulness Auditing")
    enable_llm_auditor = st.checkbox(
        "Run Deep LLM Verifier",
        value=False,
        help="Executes an additional audit pass with the LLM to verify factual claims. Increases token usage.",
    )

# Input Section
st.subheader("Enter YouTube Video URL")

youtube_url = st.text_input(
    "YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    help="Supports standard, shortened (youtu.be), embed, shorts, and timestamped URLs.",
)

generate_clicked = st.button("Generate Notes", type="primary")

if generate_clicked:
    if not youtube_url or not youtube_url.strip():
        st.warning("Please enter a valid YouTube video URL.")
        st.stop()

    # Progress Stages Container
    with st.status("Executing summarization pipeline...", expanded=True) as status:
        # Stage 1: URL Validation & Video ID Extraction
        try:
            video_id = extract_video_id(youtube_url)
            st.write("✓ URL validated")
            st.write("✓ Video ID extracted")
        except Exception as err:
            err_title, err_desc = format_user_error(err)
            status.update(label=err_title, state="error", expanded=True)
            st.error(f"**{err_title}**: {err_desc}")
            st.stop()

        # Stage 2: Zero-Download Transcript Retrieval via YouTubeTranscriptProvider
        try:
            raw_segments = fetch_cached_transcript(video_id)
            st.write("✓ Transcript retrieved")
        except (EnglishTranscriptNotFoundError, CaptionsDisabledError):
            status.update(label="Captions unavailable", state="error", expanded=True)
            st.error("English captions are unavailable for this video.")
            st.info(
                "💡 **Architectural Note (Whisper Extension):**\n\n"
                "In strict compliance with our zero-download architecture, this application does not download "
                "YouTube video or audio streams. For videos lacking native subtitles, the `WhisperFallbackProvider` "
                "architecture supports transcribing explicitly supplied local audio recordings."
            )
            st.stop()
        except Exception as err:
            err_title, err_desc = format_user_error(err)
            status.update(label=err_title, state="error", expanded=True)
            st.error(f"**{err_title}**: {err_desc}")
            st.stop()

        # Stage 3: Transcript Preprocessing
        try:
            processed = clean_transcript(raw_segments)
            clean_text = processed["clean_text"]
            if not clean_text:
                status.update(label="Empty transcript", state="error", expanded=True)
                st.error("Transcript cleaning yielded no textual content.")
                st.stop()

            st.write("✓ Transcript cleaned")
            st.write("✓ Transcript prepared")
        except Exception as err:
            err_title, err_desc = format_user_error(err)
            status.update(label=err_title, state="error", expanded=True)
            st.error(f"**{err_title}**: {err_desc}")
            st.stop()

        # Stage 4: AI Summarization via NVIDIA NIM
        st.write("✓ AI summarization in progress")
        try:
            notes_result = generate_notes(
                transcript=clean_text,
                max_words_per_chunk=int(chunk_size),
                overlap_words=int(chunk_overlap),
            )
        except Exception as err:
            err_title, err_desc = format_user_error(err)
            status.update(label=err_title, state="error", expanded=True)
            st.error(f"**{err_title}**: {err_desc}")
            st.stop()

        # Stage 5: Markdown Formatting & Validation
        try:
            formatted_markdown = format_as_markdown(notes_result)
            validate_markdown_notes(formatted_markdown)
        except Exception as err:
            err_title, err_desc = format_user_error(err)
            status.update(label=err_title, state="error", expanded=True)
            st.error(f"**{err_title}**: {err_desc}")
            st.stop()

        # Stage 6: Grounding & Faithfulness Validation
        with st.spinner("Evaluating summary grounding..."):
            validation_result = validate_summary_faithfulness(clean_text, formatted_markdown)

        llm_audit = None
        if enable_llm_auditor:
            with st.spinner("Running deep LLM faithfulness verification pass..."):
                llm_audit = verify_faithfulness_with_llm(clean_text, formatted_markdown)

        st.write("✓ Notes generated")
        status.update(label="Notes generated successfully!", state="complete", expanded=False)

    st.success("✅ Notes Generated Successfully")

    # Render Major Output Sections
    st.markdown("## Executive Summary")
    st.markdown(notes_result["executive_summary"])

    st.markdown("---")
    st.markdown("## Key Concepts")
    if notes_result["key_concepts"]:
        for concept in notes_result["key_concepts"]:
            st.markdown(f"- {concept}")
    else:
        st.markdown("- No explicit key concepts identified.")

    st.markdown("---")
    st.markdown("## Action Items")
    if notes_result["action_items"]:
        for action in notes_result["action_items"]:
            if "no explicit action items were identified" in action.lower():
                st.markdown(action)
            else:
                st.markdown(f"- {action}")
    else:
        st.markdown("No explicit action items were identified in the transcript.")

    st.markdown("---")

    # Expandable: Faithfulness & Grounding Check
    with st.expander("🛡️ Faithfulness & Grounding Check (Automated Heuristics)", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        score_percent = int(validation_result["faithfulness_score"] * 100)
        cov_percent = int(validation_result["coverage_score"] * 100)
        f_col1.metric("Faithfulness Score", f"{score_percent}%")
        f_col2.metric("Technical Term Coverage", f"{cov_percent}%")
        f_col3.metric("Unsupported Numbers Flagged", len(validation_result["unsupported_numbers"]))

        if validation_result["warnings"]:
            st.warning("**Validation Review Flags:**")
            for w in validation_result["warnings"]:
                st.markdown(f"- {w}")
        else:
            st.info("✅ Automated heuristic checks found no suspicious numerical claims or missing key terms.")

        if validation_result["unsupported_numbers"]:
            st.caption(f"Flagged numbers for manual review: `{', '.join(validation_result['unsupported_numbers'])}`")

        st.caption(f"ℹ️ **Metric Note**: {validation_result['disclaimer']}")

        if llm_audit:
            st.markdown("---")
            st.markdown("**Deep LLM Auditor Verdict:**")
            if llm_audit.get("is_faithful"):
                st.success("Auditor LLM Verdict: Verified Faithful to Source Transcript.")
            else:
                st.warning("Auditor LLM Verdict: Potential discrepancy or ungrounded statement detected.")
            with st.expander("View Full Auditor Reasoning"):
                st.text(llm_audit.get("analysis", ""))

    # Expandable: Transcript Information
    metadata = notes_result.get("metadata", {})
    total_duration = max(s["start"] + s["duration"] for s in raw_segments) if raw_segments else 0.0

    with st.expander("📊 Transcript Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Video ID:** `{video_id}`")
        col1.write(f"**Video Duration:** {format_duration(total_duration)}")
        col2.write(f"**Transcript Segments:** {processed['segment_count']:,}")
        col2.write(f"**Word Count:** {processed['word_count']:,}")
        col3.write(f"**Character Count:** {processed['character_count']:,}")
        col3.write(f"**Number of Chunks:** {metadata.get('chunk_count', 1)}")

        c_time, c_strat, c_calls = st.columns(3)
        c_time.write(f"**Processing Time:** {metadata.get('processing_time_seconds', 0.0):.2f}s")
        c_strat.write(f"**Strategy Used:** `{metadata.get('strategy', 'direct')}`")
        c_calls.write(f"**LLM Calls:** {metadata.get('llm_call_count', 1)}")

    # Expandable: Transcript Preview
    with st.expander("📝 Clean Transcript Preview", expanded=False):
        preview_chars = 1500
        preview_text = clean_text[:preview_chars] + ("..." if len(clean_text) > preview_chars else "")
        st.text_area("Clean Transcript Content:", value=preview_text, height=220, disabled=True)
        st.download_button(
            "Download Full Clean Transcript (.txt)",
            data=clean_text,
            file_name=f"transcript_{video_id}.txt",
            mime="text/plain",
        )

    # Download Section: Word Document (.docx) as primary download
    st.markdown("### 📥 Download Options")
    col_d1, col_d2 = st.columns([3, 2])

    with col_d1:
        docx_bytes = export_to_docx(
            summary_data=notes_result,
            title=f"YouTube Study Notes - {video_id}",
            metadata={
                "Video ID": video_id,
                "Duration": format_duration(total_duration),
                "Strategy": metadata.get("strategy", "direct"),
                "Source URL": youtube_url,
            },
        )
        st.download_button(
            label="📄 Download Notes as Word Document (.docx)",
            data=docx_bytes,
            file_name=f"notes_{video_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            help="Download formatted notes ready to open in Microsoft Word, Google Docs, or Apple Pages.",
        )

    with col_d2:
        st.download_button(
            label="📝 Download Markdown (.md)",
            data=formatted_markdown,
            file_name=f"notes_{video_id}.md",
            mime="text/markdown",
            help="Download raw GitHub-flavored Markdown text.",
        )


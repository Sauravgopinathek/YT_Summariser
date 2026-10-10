"""Markdown formatting and validation module.

Transforms structured summary data into clean, accessible, GitHub-flavored
Markdown without altering factual meaning or adding ungrounded information.
"""

import io
import re
from typing import Any, Dict, List, Optional, Tuple
import xml.sax.saxutils
import zipfile

from modules.exceptions import (
    MarkdownValidationError,
    MissingRequiredSectionError,
)




# Regex pattern to match emojis for removal
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff"  # Supplemental symbols & pictographs
    r"\u2600-\u27bf"           # Miscellaneous symbols
    r"\u2300-\u23ff"           # Miscellaneous technical
    r"\u2b50\u2b55\u2934\u2935"
    r"]+",
    flags=re.UNICODE,
)

# Headers required for complete, compliant notes
REQUIRED_SECTIONS = [
    "Executive Summary",
    "Key Concepts",
    "Action Items",
]


def strip_emojis(text: str) -> str:
    """Remove decorative emojis to preserve clean technical documentation style."""
    if not text:
        return ""
    return EMOJI_PATTERN.sub("", text).strip()


def clean_list_item(item: str) -> str:
    """Strip redundant leading bullets or numbered markers from an item string."""
    cleaned = item.strip()
    # Strip leading bullets: -, *, +, or numbered like 1. / 1)
    cleaned = re.sub(r"^(\s*[-*+]\s+|\s*\d+[\.\)]\s+)", "", cleaned).strip()
    return cleaned


def format_as_markdown(summary_data: Dict[str, Any], strip_decorative_emojis: bool = True) -> str:
    """Convert structured summary data into standard GitHub-flavored Markdown.

    Args:
        summary_data: Dictionary containing:
            - 'executive_summary': str
            - 'key_concepts': List[str] or str
            - 'action_items': List[str] or str
            - (optional) 'markdown': pre-existing markdown string
        strip_decorative_emojis: Whether to remove decorative emojis (default True).

    Returns:
        Formatted Markdown string.

    Raises:
        MarkdownValidationError: If input data is missing required sections or empty.
    """
    if not summary_data or not isinstance(summary_data, dict):
        raise MarkdownValidationError("Invalid summary data: expected non-empty dictionary.")

    exec_summary = summary_data.get("executive_summary", "")
    key_concepts = summary_data.get("key_concepts", [])
    action_items = summary_data.get("action_items", [])

    # If already a markdown string provided and structured fields are missing
    if not exec_summary and summary_data.get("markdown"):
        raw_md = summary_data["markdown"].strip()
        if strip_decorative_emojis:
            raw_md = strip_emojis(raw_md)
        validate_markdown_notes(raw_md)
        return raw_md

    # Format Executive Summary
    exec_summary_text = str(exec_summary).strip()
    if strip_decorative_emojis:
        exec_summary_text = strip_emojis(exec_summary_text)

    # Format Key Concepts
    concepts_lines: List[str] = []
    if isinstance(key_concepts, list):
        for c in key_concepts:
            c_text = clean_list_item(str(c))
            if strip_decorative_emojis:
                c_text = strip_emojis(c_text)
            if c_text:
                concepts_lines.append(f"- {c_text}")
    elif isinstance(key_concepts, str) and key_concepts.strip():
        raw_lines = [line.strip() for line in key_concepts.split("\n") if line.strip()]
        for line in raw_lines:
            line_cleaned = clean_list_item(line)
            if strip_decorative_emojis:
                line_cleaned = strip_emojis(line_cleaned)
            if line_cleaned:
                concepts_lines.append(f"- {line_cleaned}")

    # Format Action Items
    actions_lines: List[str] = []
    if isinstance(action_items, list):
        for idx, a in enumerate(action_items, start=1):
            a_text = clean_list_item(str(a))
            if strip_decorative_emojis:
                a_text = strip_emojis(a_text)
            if not a_text:
                continue

            # Check if action item describes a sequence step or single fallback notice
            if "no explicit action items were identified" in a_text.lower():
                actions_lines.append("No explicit action items were identified in the transcript.")
            elif re.match(r"(?i)^(step|phase|part)\s+\d+", a_text):
                actions_lines.append(f"{idx}. {a_text}")
            else:
                actions_lines.append(f"- {a_text}")

    elif isinstance(action_items, str) and action_items.strip():
        a_raw = action_items.strip()
        if strip_decorative_emojis:
            a_raw = strip_emojis(a_raw)
        if "no explicit action items were identified" in a_raw.lower():
            actions_lines.append("No explicit action items were identified in the transcript.")
        else:
            raw_lines = [line.strip() for line in a_raw.split("\n") if line.strip()]
            for line in raw_lines:
                cleaned_line = clean_list_item(line)
                if cleaned_line:
                    actions_lines.append(f"- {cleaned_line}")

    # Assemble Markdown blocks
    markdown_blocks = [
        "# Executive Summary",
        exec_summary_text,
        "",
        "# Key Concepts",
        "\n".join(concepts_lines) if concepts_lines else "- No explicit key concepts identified.",
        "",
        "# Action Items",
        "\n".join(actions_lines) if actions_lines else "No explicit action items were identified in the transcript.",
    ]

    final_markdown = "\n\n".join([b for b in markdown_blocks if b != ""]).strip()

    # Validate the final formatted output
    validate_markdown_notes(final_markdown)

    return final_markdown


def validate_markdown_notes(markdown_text: str) -> bool:
    """Verify that the Markdown text contains all three mandatory sections.

    Args:
        markdown_text: The markdown string to validate.

    Returns:
        True if all required sections exist and are non-empty.

    Raises:
        MarkdownValidationError: If any required section is absent or empty.
    """
    if not markdown_text or not isinstance(markdown_text, str) or not markdown_text.strip():
        raise MarkdownValidationError("Validation failed: Markdown text is empty.")

    missing_sections: List[str] = []

    for section in REQUIRED_SECTIONS:
        # Check for header pattern: # Section Name or ## Section Name
        header_regex = re.compile(r"(?im)^#{1,3}\s+" + re.escape(section) + r"\b")
        match = header_regex.search(markdown_text)

        if not match:
            missing_sections.append(section)
        else:
            # Check if there is content under this section
            header_end = match.end()
            next_header = re.search(r"(?m)^#{1,3}\s+", markdown_text[header_end:])
            section_content = (
                markdown_text[header_end : header_end + next_header.start()].strip()
                if next_header
                else markdown_text[header_end:].strip()
            )
            if not section_content:
                missing_sections.append(f"{section} (empty content)")

    if missing_sections:
        raise MissingRequiredSectionError(
            f"Markdown validation failed. Missing or empty required sections: {', '.join(missing_sections)}"
        )

    return True


def _parse_inline_runs(text: str) -> List[Tuple[str, bool, bool]]:
    """Parse inline Markdown **bold** and *italic* segments into text runs.

    Returns:
        List of (text_content, is_bold, is_italic) tuples.
    """
    runs: List[Tuple[str, bool, bool]] = []
    # Split on bold markers first: **...**
    bold_tokens = re.split(r"(\*\*.*?\*\*)", text)
    for token in bold_tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            runs.append((token[2:-2], True, False))
        else:
            # Check for italic markers: *...*
            italic_tokens = re.split(r"(\*.*?\*)", token)
            for itok in italic_tokens:
                if not itok:
                    continue
                if itok.startswith("*") and itok.endswith("*") and len(itok) >= 2:
                    runs.append((itok[1:-1], False, True))
                else:
                    runs.append((itok, False, False))
    return runs


def export_to_docx(
    summary_data: Dict[str, Any],
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Export structured summary notes into a Microsoft Word (.docx) binary package.

    Constructs a valid OpenXML WordprocessingML package adhering to ECMA-376
    with professional typography, bold terminology, and structured headings.

    Args:
        summary_data: Dictionary containing 'executive_summary', 'key_concepts', 'action_items'.
        title: Optional document title.
        metadata: Optional dictionary with video info (Video ID, Duration, etc.).

    Returns:
        Bytes of the complete .docx ZIP package ready for download.
    """
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>\n'
        '</Types>'
    )

    pkg_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '</Relationships>'
    )

    doc_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>\n'
        '</Relationships>'
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        '  <w:docDefaults>\n'
        '    <w:rPrDefault>\n'
        '      <w:rPr>\n'
        '        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>\n'
        '        <w:sz w:val="22"/>\n'
        '        <w:szCs w:val="22"/>\n'
        '        <w:color w:val="2D3748"/>\n'
        '      </w:rPr>\n'
        '    </w:rPrDefault>\n'
        '  </w:docDefaults>\n'
        '  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">\n'
        '    <w:name w:val="Normal"/>\n'
        '    <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr>\n'
        '  </w:style>\n'
        '  <w:style w:type="paragraph" w:styleId="Title">\n'
        '    <w:name w:val="Title"/>\n'
        '    <w:basedOn w:val="Normal"/>\n'
        '    <w:pPr><w:spacing w:before="240" w:after="200"/></w:pPr>\n'
        '    <w:rPr><w:b/><w:color w:val="1A365D"/><w:sz w:val="44"/><w:szCs w:val="44"/></w:rPr>\n'
        '  </w:style>\n'
        '  <w:style w:type="paragraph" w:styleId="Heading1">\n'
        '    <w:name w:val="heading 1"/>\n'
        '    <w:basedOn w:val="Normal"/>\n'
        '    <w:pPr><w:spacing w:before="360" w:after="140"/></w:pPr>\n'
        '    <w:rPr><w:b/><w:color w:val="2B6CB0"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>\n'
        '  </w:style>\n'
        '</w:styles>'
    )

    body_xml_parts: List[str] = []

    def make_paragraph(
        raw_text: str,
        style: Optional[str] = None,
        bold: bool = False,
        italic: bool = False,
        color: Optional[str] = None,
        space_after: int = 160,
        bullet: bool = False,
    ) -> str:
        p_pr_elements = []
        if style:
            p_pr_elements.append(f'<w:pStyle w:val="{style}"/>')
        if bullet:
            p_pr_elements.append('<w:ind w:left="720"/>')
        p_pr_elements.append(f'<w:spacing w:after="{space_after}"/>')

        p_pr_xml = f"<w:pPr>{''.join(p_pr_elements)}</w:pPr>" if p_pr_elements else ""

        # Parse inline markdown runs (e.g. bolded terms)
        runs = _parse_inline_runs(raw_text)
        runs_xml = []

        if bullet:
            runs_xml.append('<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">&#8226;  </w:t></w:r>')

        for r_text, r_bold, r_italic in runs:
            r_pr_elements = []
            if bold or r_bold:
                r_pr_elements.append("<w:b/>")
            if italic or r_italic:
                r_pr_elements.append("<w:i/>")
            if color:
                r_pr_elements.append(f'<w:color w:val="{color}"/>')

            r_pr_xml = f"<w:rPr>{''.join(r_pr_elements)}</w:rPr>" if r_pr_elements else ""
            escaped = xml.sax.saxutils.escape(r_text)
            runs_xml.append(f'<w:r>{r_pr_xml}<w:t xml:space="preserve">{escaped}</w:t></w:r>')

        return f"<w:p>{p_pr_xml}{''.join(runs_xml)}</w:p>"

    # 1. Document Title
    doc_title = title or "YouTube Video Study Notes"
    body_xml_parts.append(make_paragraph(doc_title, style="Title", bold=True))

    # 2. Metadata subtitle if present
    if metadata:
        meta_items = [f"{k}: {v}" for k, v in metadata.items() if v]
        if meta_items:
            body_xml_parts.append(
                make_paragraph(" | ".join(meta_items), italic=True, color="718096", space_after=240)
            )

    # 3. Executive Summary
    body_xml_parts.append(make_paragraph("Executive Summary", style="Heading1", bold=True))
    exec_summary = summary_data.get("executive_summary", "")
    for para in exec_summary.split("\n"):
        para_clean = para.strip()
        if para_clean:
            body_xml_parts.append(make_paragraph(para_clean))

    # 4. Key Concepts
    body_xml_parts.append(make_paragraph("Key Concepts", style="Heading1", bold=True))
    key_concepts = summary_data.get("key_concepts", [])
    if isinstance(key_concepts, list) and key_concepts:
        for concept in key_concepts:
            concept_clean = clean_list_item(concept)
            if concept_clean:
                body_xml_parts.append(make_paragraph(concept_clean, bullet=True))
    elif isinstance(key_concepts, str) and key_concepts.strip():
        for line in key_concepts.strip().split("\n"):
            line_clean = clean_list_item(line)
            if line_clean:
                body_xml_parts.append(make_paragraph(line_clean, bullet=True))
    else:
        body_xml_parts.append(make_paragraph("No explicit key concepts identified in the transcript."))

    # 5. Action Items
    body_xml_parts.append(make_paragraph("Action Items", style="Heading1", bold=True))
    action_items = summary_data.get("action_items", [])
    if isinstance(action_items, list) and action_items:
        for action in action_items:
            action_clean = clean_list_item(action)
            if action_clean:
                if "no explicit action items were identified" in action_clean.lower():
                    body_xml_parts.append(make_paragraph("No explicit action items were identified in the transcript."))
                else:
                    body_xml_parts.append(make_paragraph(action_clean, bullet=True))
    elif isinstance(action_items, str) and action_items.strip():
        for line in action_items.strip().split("\n"):
            line_clean = clean_list_item(line)
            if line_clean:
                if "no explicit action items were identified" in line_clean.lower():
                    body_xml_parts.append(make_paragraph("No explicit action items were identified in the transcript."))
                else:
                    body_xml_parts.append(make_paragraph(line_clean, bullet=True))
    else:
        body_xml_parts.append(make_paragraph("No explicit action items were identified in the transcript."))

    # 6. Page Margins (Letter size, 1 inch margins)
    body_xml_parts.append(
        '<w:sectPr>'
        '  <w:pgSz w:w="12240" w:h="15840"/>'
        '  <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        '</w:sectPr>'
    )

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        f'  <w:body>\n{"".join(body_xml_parts)}\n  </w:body>\n'
        '</w:document>'
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml)
        z.writestr("_rels/.rels", pkg_rels_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels_xml)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/document.xml", doc_xml)

    return buf.getvalue()


"""
==============================================================================
Text Cleaner — Normalize and sanitize extracted PDF text
==============================================================================

WHY TEXT CLEANING IS CRITICAL FOR RAG:
    Raw PDF text is messy. PyMuPDF does a good job, but academic papers have:
        - Hyphenated line breaks ("atten-\ntion" → "attention")
        - Excessive whitespace from column layouts
        - Headers/footers repeated on every page (page numbers, journal names)
        - Unicode artifacts (ligatures like ﬁ, ﬂ, special dashes)
        - Reference sections that bloat the index with citation noise

    If we embed this noise, retrieval quality degrades significantly.
    A clean chunk → better embedding → better retrieval → better answers.

DESIGN DECISIONS:
    - Cleaning is applied PER-PAGE (before chunking) to preserve page metadata.
    - Each cleaning step is a small, composable function for testability.
    - The `clean_text` function chains them in a sensible order.
    - Reference section removal is OPTIONAL (toggled by parameter) because
      sometimes users want to ask about cited works.

TRADEOFFS:
    - Aggressive cleaning risks removing meaningful content (e.g., URLs, emails).
      We err on the side of caution — only clean obvious noise patterns.
    - Header/footer removal uses a heuristic (short repeated lines). This works
      for ~90% of papers but may miss unusual layouts.
==============================================================================
"""

import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual cleaning functions (composable, testable)
# ---------------------------------------------------------------------------


def fix_hyphenated_line_breaks(text: str) -> str:
    """
    Rejoin words split across lines by hyphenation.

    Example: "atten-\ntion" → "attention"

    WHY: PDF extractors break lines at column/page boundaries. Hyphenated
    words at line ends are a common artifact in two-column academic papers.
    If left unfixed, "attention" becomes two separate tokens in the embedding,
    degrading semantic match quality.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/tabs into single spaces; normalize line breaks.

    WHY: Multi-column layouts produce erratic spacing. Embeddings don't care
    about extra spaces, but they waste tokens in chunks (reducing information
    density) and make prompts harder for the LLM to parse.
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse 3+ consecutive newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def fix_unicode_artifacts(text: str) -> str:
    """
    Replace common Unicode ligatures and special characters.

    WHY: Some PDFs use ligatures (ﬁ, ﬂ, ﬀ) that look identical to normal
    text but are different Unicode codepoints. If "ﬁne-tuning" is stored as
    a ligature but the user queries "fine-tuning", the embedding similarity
    drops. Normalizing these ensures consistent matching.
    """
    replacements = {
        "\ufb01": "fi",   # ﬁ ligature
        "\ufb02": "fl",   # ﬂ ligature
        "\ufb00": "ff",   # ﬀ ligature
        "\ufb03": "ffi",  # ﬃ ligature
        "\ufb04": "ffl",  # ﬄ ligature
        "\u2013": "-",    # en dash → hyphen
        "\u2014": "-",    # em dash → hyphen
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",    # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_headers_footers(text: str, max_line_length: int = 80) -> str:
    """
    Remove likely headers and footers from a page of text.

    Heuristic: Headers/footers are typically:
        - Very short lines (< max_line_length chars)
        - At the very top or bottom of the page
        - Contain page numbers, journal names, or dates

    We remove the first and last lines if they're short and look like
    header/footer material (contain digits or are very short).

    WHY: Headers like "IEEE Transactions on..." repeated on every page
    pollute chunk embeddings with irrelevant text and waste token budget.
    """
    lines = text.split("\n")

    if len(lines) < 5:
        # Too short to have meaningful headers/footers
        return text

    # Check and remove header (first 1-2 short lines)
    while lines and len(lines[0].strip()) < max_line_length:
        first = lines[0].strip()
        # Looks like a header if it's very short or contains mostly numbers
        if len(first) < 20 or re.match(r"^[\d\s\-\.]+$", first):
            lines.pop(0)
        else:
            break

    # Check and remove footer (last 1-2 short lines)
    while lines and len(lines[-1].strip()) < max_line_length:
        last = lines[-1].strip()
        if len(last) < 20 or re.match(r"^[\d\s\-\.]+$", last):
            lines.pop()
        else:
            break

    return "\n".join(lines)


def remove_references_section(text: str) -> str:
    """
    Remove the References / Bibliography section from paper text.

    WHY: Reference sections are dense lists of citations ("[1] Author, Title...")
    that produce low-quality embeddings. They rarely contain the information
    users ask about. Removing them reduces index size and improves retrieval
    precision.

    NOTE: This is OPTIONAL and applied at the full-document level, not per-page.
    Some users may want to keep references to ask "What papers does this cite?"
    """
    # Common section headers for references
    patterns = [
        r"\n\s*References\s*\n",
        r"\n\s*REFERENCES\s*\n",
        r"\n\s*Bibliography\s*\n",
        r"\n\s*BIBLIOGRAPHY\s*\n",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            logger.debug(
                f"Removing references section starting at char {match.start()}"
            )
            return text[: match.start()].strip()

    return text


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------


def clean_text(
    text: str,
    fix_hyphens: bool = True,
    fix_unicode: bool = True,
    strip_headers: bool = True,
    strip_references: bool = False,
) -> str:
    """
    Apply the full text cleaning pipeline to a raw PDF page text.

    Args:
        text: Raw text extracted from a PDF page.
        fix_hyphens: Rejoin hyphenated line breaks.
        fix_unicode: Normalize Unicode ligatures and special chars.
        strip_headers: Remove likely headers/footers.
        strip_references: Remove the References section (use on full-doc text).

    Returns:
        Cleaned text string.
    """
    if not text or not text.strip():
        return ""

    if fix_hyphens:
        text = fix_hyphenated_line_breaks(text)

    if fix_unicode:
        text = fix_unicode_artifacts(text)

    if strip_headers:
        text = remove_headers_footers(text)

    # Whitespace normalization always runs (it's non-destructive)
    text = normalize_whitespace(text)

    if strip_references:
        text = remove_references_section(text)

    return text


def clean_documents(
    documents: list[Document],
    strip_references: bool = False,
) -> list[Document]:
    """
    Clean all extracted page documents in-place.

    Applies the full cleaning pipeline to each document's page_content
    while preserving metadata. Removes documents that become empty
    after cleaning.

    Args:
        documents: List of Documents from pdf_loader.
        strip_references: Whether to remove references sections.

    Returns:
        List of cleaned Document objects (empty pages removed).
    """
    cleaned = []

    for doc in documents:
        clean_content = clean_text(
            doc.page_content,
            strip_references=strip_references,
        )

        # Skip pages that became empty after cleaning
        if len(clean_content.strip()) < 30:
            logger.debug(
                f"Dropping page {doc.metadata.get('page', '?')} — "
                f"empty after cleaning"
            )
            continue

        cleaned.append(
            Document(
                page_content=clean_content,
                metadata=doc.metadata,
            )
        )

    logger.info(
        f"Cleaned {len(documents)} pages → {len(cleaned)} retained "
        f"({len(documents) - len(cleaned)} dropped)"
    )

    return cleaned

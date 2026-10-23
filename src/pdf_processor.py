"""PDF Processing & Document Chunking Module for Designs Law Assistant.

Extracts text from the 87-page Manual of Designs Practice & Procedure PDF,
detects canonical Chapter and Section headings, preserves 1-indexed page numbers,
and chunks content while attaching rich structural metadata for RAG citations.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Deterministic chapter map based on the official Index of the Manual (87 pages)
CHAPTER_PAGE_RANGES = [
    (1, 2, "Title & Table of Contents"),
    (3, 4, "Preface"),
    (5, 6, "Chapter 1 – Introduction"),
    (7, 10, "Chapter 2 – Important definitions"),
    (11, 23, "Chapter 3 – Filing of Design Application"),
    (24, 33, "Chapter 4 – Examination and registration of designs"),
    (34, 34, "Chapter 5 – The Patent Office Journal"),
    (35, 36, "Chapter 6 – Piracy of registered designs and consequences thereof"),
    (37, 44, "Chapter 7 – Post registration procedures"),
    (45, 46, "Chapter 8 – Transfer of rights"),
    (47, 49, "Chapter 9 – General Services"),
    (50, 50, "Chapter 10 – Use & acquisition of registered design by Government"),
    (51, 53, "Chapter 11 – General powers of Controller"),
    (54, 57, "Chapter 12 – Evidences etc."),
    (58, 59, "Chapter 13 – Restrictive conditions in contracts etc."),
    (60, 60, "Chapter 14 – Appeals"),
    (61, 62, "Chapter 15 – Requirements before delivery on sales"),
    (63, 64, "Chapter 16 – Timelines"),
    (65, 85, "Annexure I – Important Case Laws"),
    (86, 87, "Annexure II – Specimen novelty statements & disclaimers in representation sheet"),
]


@dataclass
class PageData:
    page_number: int
    raw_text: str
    cleaned_text: str
    chapter: str
    sections: List[str]


def clean_page_text(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving semantic structure."""
    if not text:
        return ""
    # Normalize unicode spaces & punctuation
    text = text.replace('\xa0', ' ').replace('\u2013', '-').replace('\u2014', '-')
    # Normalize multiple line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def get_canonical_chapter(page_num: int) -> str:
    """Return the official chapter title for a given 1-indexed page number."""
    for start, end, chapter_title in CHAPTER_PAGE_RANGES:
        if start <= page_num <= end:
            return chapter_title
    return "Manual of Designs Practice & Procedure"


def extract_sections_from_page(text: str) -> List[str]:
    """Find section identifiers like 03.06.02.06 Classification of designs, 03.08 Fee, 06.04 Period of protection."""
    # Match patterns like:
    # 02.02.01 Controller
    # 03.06.02.06 Classification of designs
    # 03.08 Fee
    # 06.04 Period of protection
    # 07.02 Extension of copyright (renewal)
    section_pattern = r'(?:^|\n)\s*(\d{2}(?:\.\d{2})+(?:\.[A-Za-z0-9]+)*)\s+([A-Za-z0-9 ,/–\(\)\-]+)'
    found = []
    for match in re.finditer(section_pattern, text):
        sec_num = match.group(1).strip()
        sec_title = match.group(2).strip()
        found.append(f"Section {sec_num}: {sec_title}")
    return found


def load_and_parse_pdf(pdf_path: str) -> List[PageData]:
    """Parse the PDF page-by-page preserving accurate page numbers and headings."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_data: List[PageData] = []

    for i in range(total_pages):
        page_num = i + 1  # 1-indexed to match PDF viewers exactly
        page = reader.pages[i]
        raw_text = page.extract_text() or ""
        cleaned = clean_page_text(raw_text)

        chapter = get_canonical_chapter(page_num)
        sections = extract_sections_from_page(cleaned)

        pages_data.append(
            PageData(
                page_number=page_num,
                raw_text=raw_text,
                cleaned_text=cleaned,
                chapter=chapter,
                sections=sections,
            )
        )

    return pages_data


def chunk_manual_pages(
    pages_data: List[PageData],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    source_filename: str = "Manual of Designs Practice & Procedure"
) -> List[Document]:
    """Convert parsed pages into LangChain Document chunks with rich metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents: List[Document] = []
    last_known_section = "General Provisions"

    for page in pages_data:
        if not page.cleaned_text.strip():
            continue

        # If new sections are found on this page, update last known section
        if page.sections:
            last_known_section = page.sections[0]
        elif page.page_number <= 4:
            last_known_section = "Preface / Preliminary"

        raw_chunks = splitter.split_text(page.cleaned_text)

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            # Check if this specific chunk contains a section header
            chunk_sections = extract_sections_from_page(chunk_text)
            if chunk_sections:
                current_chunk_section = chunk_sections[0]
                last_known_section = chunk_sections[-1]
            else:
                current_chunk_section = last_known_section

            # Construct clean human-readable citation label
            if "Preface" in page.chapter or "Title" in page.chapter:
                citation_label = f"{page.chapter} (PDF Page {page.page_number})"
            else:
                citation_label = f"{page.chapter}, {current_chunk_section} (PDF Page {page.page_number})"

            metadata = {
                "source": source_filename,
                "page": page.page_number,
                "chapter": page.chapter,
                "section": current_chunk_section,
                "citation": citation_label,
                "chunk_index": chunk_idx,
            }

            # Prepend contextual heading for enhanced semantic retrieval
            contextual_header = (
                f"[{source_filename} | {page.chapter} | {current_chunk_section} | PDF Page {page.page_number}]\n"
            )
            content_with_context = contextual_header + chunk_text

            doc = Document(
                page_content=content_with_context,
                metadata=metadata
            )
            documents.append(doc)

    return documents

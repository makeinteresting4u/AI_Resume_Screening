from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Dict

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noisy characters while preserving meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyPDF2 is required to parse PDF files.") from exc
    reader = PdfReader(file_obj)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def extract_text_from_docx(file_obj: BinaryIO) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("python-docx is required to parse DOCX files.") from exc
    document = Document(file_obj)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return clean_text("\n".join(paragraphs))


def extract_text_from_txt(file_obj: BinaryIO) -> str:
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    return clean_text(content)


def parse_resume_file(filename: str, file_obj: BinaryIO) -> Dict[str, str]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        text = extract_text_from_pdf(file_obj)
    elif extension == ".docx":
        text = extract_text_from_docx(file_obj)
    else:
        text = extract_text_from_txt(file_obj)

    if not text:
        raise ValueError("No readable text found in the uploaded resume.")

    return {
        "candidate_name": infer_candidate_name(filename, text),
        "text": text,
    }


def infer_candidate_name(filename: str, text: str) -> str:
    """
    Best-effort candidate name inference.
    Falls back to the file name when the resume structure is ambiguous.
    """
    first_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in first_lines[:5]:
        if 2 <= len(line.split()) <= 4 and len(line) < 60:
            if not any(char.isdigit() for char in line):
                return line.title()
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def extract_skills(text: str) -> list[str]:
    """Simple keyword-based skill extraction for highlighting and gap analysis."""
    skill_dictionary = {
        "python", "sql", "machine learning", "deep learning", "nlp", "tensorflow",
        "pytorch", "fastapi", "streamlit", "aws", "azure", "docker", "kubernetes",
        "pandas", "numpy", "scikit-learn", "faiss", "langchain", "llm", "openai",
        "data analysis", "statistics", "power bi", "tableau", "spark", "git",
        "rest api", "microservices", "etl", "postgresql", "mongodb"
    }
    lowered = text.lower()
    found = [skill for skill in skill_dictionary if skill in lowered]
    return sorted(found)

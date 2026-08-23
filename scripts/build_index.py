from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "source"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def chunk_document(
    document: dict[str, Any],
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    title = str(document.get("title", "Untitled"))
    url = str(document.get("url", ""))
    parts = [
        _normalize_text(str(section.get("text", "")))
        for section in document.get("sections", [])
        if _normalize_text(str(section.get("text", "")))
    ]
    text = "\n".join(parts).strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    sequence = 1
    slug = _slugify(title)

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{slug}-{sequence:04d}",
                    "source_title": title,
                    "source_url": url,
                    "text": chunk_text,
                }
            )
            sequence += 1

        if end >= len(text):
            break
        start = end - overlap_chars

    return chunks


def build_index(raw_dir: Path, index_path: Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    index_path = Path(index_path)

    documents: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        documents.append(json.loads(path.read_text(encoding="utf-8")))

    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks.extend(chunk_document(document))

    if not chunks:
        raise ValueError("No text chunks were found in the raw dataset")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=9000,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)

    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, index_path)

    return {
        "sources": len(documents),
        "chunks": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "index_path": str(index_path),
    }


if __name__ == "__main__":
    summary = build_index(Path("data/raw"), Path("data/index/rag_index.pkl"))
    print(json.dumps(summary, indent=2))

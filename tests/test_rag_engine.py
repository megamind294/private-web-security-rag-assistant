from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_engine import RAGEngine


def _write_test_index(path: Path) -> None:
    chunks = [
        {
            "chunk_id": "sql-injection-prevention-0001",
            "source_title": "SQL Injection Prevention",
            "source_url": "https://example.test/sql",
            "text": "Use prepared statements and parameterized queries to prevent SQL injection.",
        },
        {
            "chunk_id": "session-management-0001",
            "source_title": "Session Management",
            "source_url": "https://example.test/session",
            "text": "Use secure, HttpOnly, and SameSite cookie attributes for sessions.",
        },
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, path)


def test_retrieve_ranks_parameterized_query_chunk_first(tmp_path):
    index_path = tmp_path / "rag_index.pkl"
    _write_test_index(index_path)
    engine = RAGEngine(index_path)

    results = engine.retrieve("How do parameterized queries stop SQL injection?", top_k=2)

    assert results[0]["source_title"] == "SQL Injection Prevention"
    assert results[0]["score"] > results[1]["score"]


def test_security_answer_uses_extractive_fallback_without_providers(tmp_path, monkeypatch):
    index_path = tmp_path / "rag_index.pkl"
    _write_test_index(index_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    engine = RAGEngine(index_path)

    result = engine.answer("How should I prevent SQL injection?", mode="security")

    assert result["mode"] == "extractive"
    assert result["answer"]
    assert result["citations"][0]["source_title"] == "SQL Injection Prevention"
    assert 0.0 <= result["citations"][0]["score"] <= 1.0
    assert {"source_title", "source_url", "chunk_id", "score"} <= set(result["citations"][0])

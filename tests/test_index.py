from scripts.build_index import chunk_document


def test_chunk_document_preserves_source_metadata():
    document = {
        "title": "SQL Injection Prevention",
        "url": "https://example.test/sql",
        "sections": [
            {"type": "heading", "text": "Prepared Statements"},
            {"type": "paragraph", "text": "Use parameterized queries."},
        ],
    }

    chunks = chunk_document(document, max_chars=200, overlap_chars=20)

    assert len(chunks) == 1
    assert chunks[0]["source_title"] == "SQL Injection Prevention"
    assert chunks[0]["source_url"] == "https://example.test/sql"
    assert "Prepared Statements" in chunks[0]["text"]
    assert "parameterized queries" in chunks[0]["text"]
    assert chunks[0]["chunk_id"] == "sql-injection-prevention-0001"

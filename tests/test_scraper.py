from scripts.scrape_owasp import extract_page_text


def test_extract_page_text_keeps_headings_and_paragraphs():
    html = """
    <html><body>
      <nav>skip me</nav>
      <h1>CSRF Prevention</h1>
      <p>Use anti-CSRF tokens.</p>
      <ul><li>Prefer SameSite cookies.</li></ul>
      <script>ignore()</script>
    </body></html>
    """

    sections = extract_page_text(html)

    assert sections == [
        {"type": "heading", "text": "CSRF Prevention"},
        {"type": "paragraph", "text": "Use anti-CSRF tokens."},
        {"type": "list_item", "text": "Prefer SameSite cookies."},
    ]

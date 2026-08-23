# Private Web Security RAG Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct and polish the university Private Web Security Assistant into a working public GitHub project that answers web-security questions using a locally indexed OWASP knowledge base with citations and a reliable local fallback.

**Architecture:** A scraper collects a fixed set of OWASP Cheat Sheet pages into local JSON, an indexing module cleans and chunks the text and builds a TF-IDF matrix, and a retrieval engine ranks chunks with cosine similarity. A Flask application exposes `/`, `/health`, and `/api/ask`, with answer generation supporting extractive local responses by default and optional OpenAI or Ollama generation when configured.

**Tech Stack:** Python 3.11+, Flask, requests, BeautifulSoup4, scikit-learn, joblib, pytest, HTML/CSS, optional OpenAI Responses API, optional Ollama HTTP API.

**Spec:** `megamind294/megamind294/docs/superpowers/specs/2026-08-23-university-github-portfolio-design.md`

## Global Constraints

- Preserve the university project's documented architecture: OWASP scraping, preprocessing/chunking, TF-IDF indexing, cosine-similarity retrieval, Flask UI/API, citations, optional OpenAI/Ollama generation, and extractive fallback.
- The knowledge base covers six OWASP topics: CSRF, XSS, SQL Injection, Authentication, Session Management, and Authorization.
- Do not commit API keys, `.env`, credentials, personal IDs, grades, or private university data.
- Use `.env.example` for optional provider configuration and keep `.env` ignored.
- The README must disclose that this repository is a reconstructed and polished version of university coursework.
- Do not fabricate benchmark values during reconstruction; only report measurements actually produced by the rebuilt code.
- Default execution must remain functional without OpenAI or Ollama.

---

## File Structure

```text
private-web-security-rag-assistant/
├── app.py                         # Flask app and HTTP routes
├── rag_engine.py                  # retrieval + answer generation orchestration
├── requirements.txt               # runtime/test dependencies
├── .env.example                   # optional provider settings
├── .gitignore                     # secrets/cache/index exclusions
├── README.md                      # recruiter-friendly project documentation
├── scripts/
│   ├── __init__.py
│   ├── scrape_owasp.py            # fetch and normalize selected OWASP pages
│   └── build_index.py             # chunk text and persist TF-IDF index
├── data/
│   ├── raw/.gitkeep               # local scraped JSON destination
│   └── index/.gitkeep             # local serialized index destination
├── templates/
│   └── index.html                 # browser UI
├── static/
│   └── style.css                  # UI styling
└── tests/
    ├── test_scraper.py            # source extraction tests
    ├── test_index.py              # chunk/index tests
    ├── test_rag_engine.py         # retrieval/answer fallback tests
    └── test_app.py                # Flask route/API tests
```

---

### Task 1: Project foundation and source model

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `scripts/__init__.py`
- Create: `scripts/scrape_owasp.py`
- Create: `data/raw/.gitkeep`
- Create: `data/index/.gitkeep`
- Create: `tests/test_scraper.py`

**Interfaces:**
- Consumes: none.
- Produces: `OWASP_SOURCES: dict[str, str]`, `extract_page_text(html: str) -> list[dict[str, str]]`, `scrape_source(title: str, url: str) -> dict`, `scrape_all(output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Add dependency and secret-handling files**

Create `requirements.txt` with:

```text
Flask>=3.0,<4.0
requests>=2.32,<3.0
beautifulsoup4>=4.12,<5.0
scikit-learn>=1.5,<2.0
joblib>=1.4,<2.0
python-dotenv>=1.0,<2.0
pytest>=8.0,<9.0
```

Create `.gitignore` with:

```text
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
data/raw/*.json
data/index/*.pkl
.DS_Store
```

Create `.env.example` with:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

- [ ] **Step 2: Write scraper tests before implementation**

Create `tests/test_scraper.py`:

```python
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
```

- [ ] **Step 3: Run scraper test and verify failure**

Run:

```bash
pytest tests/test_scraper.py -v
```

Expected: collection/import failure because `scripts.scrape_owasp` does not yet exist.

- [ ] **Step 4: Implement fixed OWASP source registry and extraction**

Create `scripts/scrape_owasp.py` with the six documented OWASP Cheat Sheet URLs and an `extract_page_text` implementation that removes `script`, `style`, `nav`, `footer`, and `header` elements, then emits normalized `h1`-`h4`, `p`, and `li` text records.

Use this public interface:

```python
OWASP_SOURCES = {
    "Cross-Site Request Forgery Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html",
    "Cross Site Scripting Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    "SQL Injection Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
    "Authentication": "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
    "Session Management": "https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html",
    "Authorization": "https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html",
}
```

`extract_page_text(html)` must return `list[dict[str, str]]` with keys `type` and `text`.

`scrape_source(title, url)` must return:

```python
{
    "title": title,
    "url": url,
    "sections": [...],
}
```

`scrape_all(output_dir)` must write one UTF-8 JSON file per source and return the written paths.

- [ ] **Step 5: Run scraper tests**

Run:

```bash
pytest tests/test_scraper.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit foundation**

```bash
git add .gitignore .env.example requirements.txt scripts data tests/test_scraper.py
git commit -m "feat: add OWASP scraper foundation"
```

---

### Task 2: Text chunking and TF-IDF index builder

**Files:**
- Create: `scripts/build_index.py`
- Create: `tests/test_index.py`

**Interfaces:**
- Consumes: scraper JSON records with `title`, `url`, and `sections`.
- Produces: `chunk_document(document: dict, max_chars: int = 1200, overlap_chars: int = 200) -> list[dict]`, `build_index(raw_dir: Path, index_path: Path) -> dict`.

- [ ] **Step 1: Write chunking test**

Create `tests/test_index.py`:

```python
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
```

- [ ] **Step 2: Run index test and verify failure**

```bash
pytest tests/test_index.py -v
```

Expected: import failure because `scripts.build_index` does not exist.

- [ ] **Step 3: Implement deterministic chunking**

Create `scripts/build_index.py`. Concatenate normalized section text, produce overlapping character windows no larger than `max_chars`, and attach these keys to every chunk:

```python
{
    "chunk_id": str,
    "source_title": str,
    "source_url": str,
    "text": str,
}
```

Chunk IDs must use a lowercase slug derived from the title plus a 4-digit sequence, for example `sql-injection-prevention-0001`.

- [ ] **Step 4: Implement TF-IDF persistence**

`build_index(raw_dir, index_path)` must:

1. Load all `*.json` scraper files from `raw_dir`.
2. Build all chunks with `chunk_document`.
3. Fit `sklearn.feature_extraction.text.TfidfVectorizer` with `ngram_range=(1, 2)`, `max_features=9000`, and English stop words.
4. Transform chunk text into a sparse matrix.
5. Persist one joblib dictionary to `index_path`:

```python
{
    "vectorizer": vectorizer,
    "matrix": matrix,
    "chunks": chunks,
}
```

6. Return a summary dictionary containing `sources`, `chunks`, `vocabulary_size`, and `index_path`.

- [ ] **Step 5: Run index tests**

```bash
pytest tests/test_index.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit index builder**

```bash
git add scripts/build_index.py tests/test_index.py
git commit -m "feat: add TF-IDF index builder"
```

---

### Task 3: Retrieval engine and local extractive answers

**Files:**
- Create: `rag_engine.py`
- Create: `tests/test_rag_engine.py`

**Interfaces:**
- Consumes: joblib index dictionary from Task 2.
- Produces: `RAGEngine(index_path: Path)`, `RAGEngine.retrieve(question: str, top_k: int = 4) -> list[dict]`, `RAGEngine.answer(question: str, mode: str = "security") -> dict`.

- [ ] **Step 1: Write retrieval and fallback tests**

Create `tests/test_rag_engine.py` using a temporary two-document TF-IDF index. Assert that a question about parameterized queries ranks the SQL Injection chunk first and that `answer(..., mode="security")` returns `mode == "extractive"`, a non-empty `answer`, and citations containing `source_title`, `source_url`, `chunk_id`, and `score`.

Use this assertion shape:

```python
assert result["mode"] == "extractive"
assert result["answer"]
assert result["citations"][0]["source_title"] == "SQL Injection Prevention"
assert 0.0 <= result["citations"][0]["score"] <= 1.0
```

- [ ] **Step 2: Run engine test and verify failure**

```bash
pytest tests/test_rag_engine.py -v
```

Expected: import failure because `rag_engine.py` does not exist.

- [ ] **Step 3: Implement `RAGEngine.retrieve`**

Load the joblib payload in `__init__`. For retrieval, transform the question with the saved vectorizer, compute cosine similarity against the saved matrix, sort descending, and return up to `top_k` chunk dictionaries augmented with numeric `score`.

Reject blank questions with:

```python
raise ValueError("Question must not be empty")
```

- [ ] **Step 4: Implement extractive answer generation**

For local fallback, compose a concise answer from the highest-ranked retrieved chunk texts. Return:

```python
{
    "question": question,
    "mode": "extractive",
    "answer": answer_text,
    "citations": [
        {
            "source_title": str,
            "source_url": str,
            "chunk_id": str,
            "score": float,
        }
    ],
}
```

If every retrieval score is zero, return a useful message that the local knowledge base did not find a confident match while still returning the ranked citations.

- [ ] **Step 5: Add optional OpenAI and Ollama adapters**

Implement private methods `_answer_with_openai(question, retrieved)` and `_answer_with_ollama(question, retrieved)`.

Provider selection rules:

- `mode="security"`: prefer OpenAI only when `OPENAI_API_KEY` exists; otherwise try Ollama if `OLLAMA_BASE_URL` and `OLLAMA_MODEL` are configured and reachable; otherwise use extractive fallback.
- `mode="local"`: use Ollama if reachable, otherwise extractive fallback.
- `mode="extractive"`: always use extractive fallback.
- The system prompt must instruct the generator to answer only from supplied OWASP context and not invent citations.
- Provider failures must fall back to extractive mode instead of crashing the request.

- [ ] **Step 6: Run engine tests**

```bash
pytest tests/test_rag_engine.py -v
```

Expected: PASS without requiring OpenAI or Ollama.

- [ ] **Step 7: Commit retrieval engine**

```bash
git add rag_engine.py tests/test_rag_engine.py
git commit -m "feat: add RAG retrieval and fallback answers"
```

---

### Task 4: Flask API and health checks

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `RAGEngine` from Task 3.
- Produces: Flask routes `GET /`, `GET /health`, and `POST /api/ask`.

- [ ] **Step 1: Write API tests**

Create `tests/test_app.py` with a fake engine injected through `create_app(engine=...)`.

Required assertions:

```python
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ask_rejects_blank_question(client):
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question must not be empty"
```

Also verify a valid request returns the fake engine payload with HTTP 200.

- [ ] **Step 2: Run API tests and verify failure**

```bash
pytest tests/test_app.py -v
```

Expected: import failure because `app.py` does not exist.

- [ ] **Step 3: Implement Flask application factory**

Create:

```python
def create_app(engine=None):
    ...
```

Behavior:

- `GET /health` returns JSON containing `status: "ok"` and whether the RAG index is available.
- `POST /api/ask` expects JSON `{ "question": str, "mode": str | omitted }`.
- Blank or missing `question` returns HTTP 400 with `{ "error": "Question must not be empty" }`.
- Valid requests call `engine.answer(question, mode)` and return JSON.
- Unexpected errors are logged server-side and return HTTP 500 with a generic error message that does not expose secrets or stack traces.

- [ ] **Step 4: Run API tests**

```bash
pytest tests/test_app.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Flask API**

```bash
git add app.py tests/test_app.py
git commit -m "feat: expose RAG assistant Flask API"
```

---

### Task 5: Browser UI

**Files:**
- Create: `templates/index.html`
- Create: `static/style.css`
- Modify: `app.py`
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `POST /api/ask`.
- Produces: responsive browser UI with question input, mode selector, loading state, answer text, and source citations.

- [ ] **Step 1: Extend route test for HTML UI**

Add:

```python
def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Web Security Assistant" in response.data
    assert b"Ask a security question" in response.data
```

- [ ] **Step 2: Run UI route test and verify failure**

```bash
pytest tests/test_app.py::test_index_page_renders -v
```

Expected: FAIL because `/` is not yet implemented or does not contain the required content.

- [ ] **Step 3: Implement UI template**

`templates/index.html` must include:

- Project title: `Private Web Security Assistant`.
- Short explanation that answers are grounded in an OWASP knowledge base.
- Question textarea/input with visible label `Ask a security question`.
- Mode selector with `Security RAG`, `Local`, and `Extractive` options.
- Submit button.
- Answer panel.
- Citation list showing source title, similarity score, and clickable OWASP URL.
- Client-side `fetch('/api/ask', ...)` logic with loading and error states.

Do not embed API keys or provider secrets in the HTML/JavaScript.

- [ ] **Step 4: Add clean responsive styling**

`static/style.css` must style the page as a professional developer-project demo: centered max-width layout, readable typography, responsive form controls, distinct answer/source cards, keyboard-visible focus states, and usable mobile spacing.

- [ ] **Step 5: Render template from `/` and run tests**

Update `app.py` so `GET /` calls `render_template("index.html")`.

Run:

```bash
pytest tests/test_app.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit UI**

```bash
git add templates/index.html static/style.css app.py tests/test_app.py
git commit -m "feat: add web interface for security assistant"
```

---

### Task 6: End-to-end data build, validation, and documentation

**Files:**
- Modify: `README.md`
- Modify: `scripts/scrape_owasp.py`
- Modify: `scripts/build_index.py`
- Modify: `app.py`
- Create: `docs/architecture.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reproducible setup/build/run workflow and verified repository documentation.

- [ ] **Step 1: Add command-line entry points**

Ensure these commands work from repository root:

```bash
python scripts/scrape_owasp.py
python scripts/build_index.py
python app.py
```

Default locations:

- scraper output: `data/raw/`
- serialized index: `data/index/rag_index.pkl`
- Flask host: `127.0.0.1`
- Flask port: `5000`

- [ ] **Step 2: Build the real local dataset and index**

Run:

```bash
python scripts/scrape_owasp.py
python scripts/build_index.py
```

Verify the build summary reports exactly six source documents. Record the actual reconstructed chunk count and vocabulary size from this run; do not force them to equal historical report values if the current OWASP pages have changed.

- [ ] **Step 3: Run the full automated test suite**

```bash
pytest -v
```

Expected: all tests PASS with no OpenAI key and no running Ollama instance.

- [ ] **Step 4: Smoke-test the HTTP API**

Start:

```bash
python app.py
```

Then verify:

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I prevent CSRF in a Flask application?","mode":"extractive"}'
```

Expected: health status `ok`; answer payload includes non-empty `answer` and at least one OWASP citation.

- [ ] **Step 5: Write architecture documentation**

Create `docs/architecture.md` describing this concrete flow:

```text
OWASP Cheat Sheet pages
        ↓
requests + BeautifulSoup scraper
        ↓
normalized JSON documents
        ↓
cleaning + overlapping chunks
        ↓
TF-IDF vectorizer + sparse matrix
        ↓
cosine similarity top-k retrieval
        ↓
OpenAI / Ollama / extractive answer generation
        ↓
Flask API + browser UI + citations
```

Explain what data is persisted locally and what happens when optional providers are unavailable.

- [ ] **Step 6: Replace placeholder README with recruiter-ready documentation**

README sections must be:

1. Project title and one-paragraph summary.
2. `University Coursework Reconstruction` disclosure explaining that the current repository was reconstructed and polished from the original university project/report.
3. Features.
4. Architecture.
5. Tech stack.
6. Knowledge-base topics.
7. Setup.
8. Build the OWASP index.
9. Run the app.
10. API example.
11. Run tests.
12. Optional OpenAI/Ollama configuration.
13. Security/privacy notes.
14. Actual reconstructed results: source count, chunk count, vocabulary size, and measured local latency only if measured in the current build.
15. Future improvements: dense embeddings, FAISS/ChromaDB, PDF/class-note ingestion, conversation memory, authentication, and retrieval-score visualization.

- [ ] **Step 7: Final verification**

Run:

```bash
pytest -v
python scripts/build_index.py
```

Then check that `git status --short` contains no `.env`, API key files, raw scraped JSON, or `rag_index.pkl` staged for commit.

- [ ] **Step 8: Commit documentation and verified project state**

```bash
git add README.md docs/architecture.md app.py scripts/build_index.py scripts/scrape_owasp.py
git commit -m "docs: complete reconstructed university RAG project"
```

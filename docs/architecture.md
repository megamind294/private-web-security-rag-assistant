# Architecture

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

## Ingestion

`scripts/scrape_owasp.py` downloads six selected OWASP Cheat Sheet pages and extracts headings, paragraphs, and list items while removing navigation, scripts, styles, headers, and footers. Each source is written as normalized JSON under `data/raw/`.

Raw scraped JSON is intentionally gitignored. Rebuilding it keeps the repository small and avoids treating a snapshot of third-party documentation as project source code.

## Indexing

`scripts/build_index.py` combines normalized sections into overlapping text chunks. Every chunk keeps its source title, source URL, and deterministic chunk ID.

`TfidfVectorizer` uses English stop words, unigrams/bigrams, and a maximum vocabulary size of 9,000 features. The vectorizer, sparse TF-IDF matrix, and chunk metadata are serialized together to `data/index/rag_index.pkl`, which is also gitignored.

## Retrieval

`RAGEngine.retrieve()` transforms the question with the saved vectorizer, computes cosine similarity against the chunk matrix, sorts by score, and returns the highest-ranked chunks with source metadata.

## Answer modes

- **Security RAG**: use OpenAI when configured; otherwise attempt local Ollama; otherwise fall back to extractive answers.
- **Local**: prefer Ollama and fall back to extractive mode if unavailable.
- **Extractive**: answer directly from the retrieved OWASP text without a generation provider.

Provider failures are designed to fall back rather than make the whole assistant unusable.

## Application layer

Flask exposes:

- `GET /` — browser UI;
- `GET /health` — service/index readiness;
- `POST /api/ask` — JSON question/answer endpoint.

The browser receives only answer text and citation metadata. API keys stay server-side in environment variables.

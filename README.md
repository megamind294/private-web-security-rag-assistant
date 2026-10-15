# Private Web Security Assistant — Local RAG

A web-security question-answering assistant built around a **local OWASP knowledge base**, TF-IDF retrieval, cosine similarity, source citations, a Flask API, and optional OpenAI/Ollama answer generation.

## University Coursework Reconstruction

This repository is a reconstructed and polished version of a university project documented in the saved project report. The recovered report describes a Private Web Security Assistant using local retrieval-augmented generation over OWASP material. The current repository rebuilds that architecture into a clean, runnable portfolio project; it is not presented as an untouched copy of every original source file.

## Features

- Six OWASP Cheat Sheet knowledge-base topics.
- Repeatable scraper that stores normalized local JSON.
- Overlapping text chunking with source metadata.
- TF-IDF vectorization with unigram/bigram features.
- Cosine-similarity top-k retrieval.
- Source title, URL, chunk ID, and similarity score citations.
- Fully local extractive fallback that needs no API key or local LLM.
- Optional OpenAI generation when `OPENAI_API_KEY` is configured.
- Optional Ollama generation for local LLM workflows.
- Flask browser UI, `/health`, and JSON `/api/ask` endpoint.
- pytest coverage for scraping, indexing, retrieval, fallback answers, and Flask routes.

## Architecture

```text
OWASP Cheat Sheet pages
        ↓
requests + BeautifulSoup
        ↓
normalized JSON
        ↓
overlapping text chunks
        ↓
TF-IDF + sparse matrix
        ↓
cosine-similarity retrieval
        ↓
OpenAI / Ollama / extractive fallback
        ↓
Flask API + browser UI + citations
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed flow.

## Tech stack

- Python 3.11+
- Flask
- Requests + BeautifulSoup
- scikit-learn
- joblib
- pytest
- HTML/CSS/JavaScript
- optional OpenAI Responses API
- optional Ollama HTTP API

## Knowledge-base topics

The rebuilt source registry covers:

1. Cross-Site Request Forgery (CSRF) Prevention
2. Cross Site Scripting (XSS) Prevention
3. SQL Injection Prevention
4. Authentication
5. Session Management
6. Authorization

The original university report also documented six source documents. Because OWASP pages change over time, current chunk counts and retrieved text can differ when the index is rebuilt.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build the OWASP index

```bash
python scripts/scrape_owasp.py
python scripts/build_index.py
```

Generated artifacts are stored under `data/raw/` and `data/index/` and are intentionally excluded from Git.

## Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## API example

```bash
curl -X POST http://127.0.0.1:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I prevent SQL injection?","mode":"extractive"}'
```

The response includes `answer`, `mode`, and `citations`.

## Answer modes

- `security` — prefer configured OpenAI, then reachable Ollama, then extractive fallback.
- `local` — prefer reachable Ollama, then extractive fallback.
- `extractive` — always answer directly from retrieved OWASP chunks.

## Optional provider configuration

Copy the variable names from `.env.example` into your own local `.env` or shell environment. Never commit API keys.

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

The default extractive mode remains usable without either provider.

## Tests

```bash
python -m pytest -q
```

The automated suite uses synthetic/test HTML and temporary TF-IDF indexes, so CI does not need to scrape live OWASP pages or use external AI providers.

## Security and privacy notes

- No API keys or `.env` secrets are committed.
- Raw scraped documents and serialized indexes are rebuilt locally.
- The generator prompt is constrained to retrieved security context.
- Provider failures fall back to local extraction.
- This project is an educational security assistant; it should not replace professional security review or authoritative OWASP guidance.

## Reconstruction results

The recovered university report documented a six-document OWASP dataset and a TF-IDF retrieval architecture. This rebuilt repository deliberately does **not** hard-code historical chunk counts or latency as current benchmarks: rebuilding against today's OWASP pages can produce different values. `scripts/build_index.py` prints the source count, current chunk count, and current vocabulary size after each build.

## Future improvements

- Dense embeddings and hybrid retrieval
- FAISS or ChromaDB persistence
- PDF/class-note ingestion
- Conversation memory
- Authentication for private deployments
- Retrieval-score visualization and evaluation datasets

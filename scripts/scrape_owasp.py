from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


OWASP_SOURCES: dict[str, str] = {
    "Cross-Site Request Forgery Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html",
    "Cross Site Scripting Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    "SQL Injection Prevention": "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
    "Authentication": "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",
    "Session Management": "https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html",
    "Authorization": "https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_page_text(html: str) -> list[dict[str, str]]:
    """Extract useful OWASP content while dropping navigation and executable markup."""
    soup = BeautifulSoup(html, "html.parser")

    for element in soup.find_all(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    sections: list[dict[str, str]] = []
    type_by_tag = {
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "p": "paragraph",
        "li": "list_item",
    }

    for element in soup.find_all(list(type_by_tag)):
        text = _normalize_text(element.get_text(" ", strip=True))
        if text:
            sections.append({"type": type_by_tag[element.name], "text": text})

    return sections


def scrape_source(title: str, url: str) -> dict[str, Any]:
    """Download one OWASP cheat sheet and return normalized structured content."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "private-web-security-rag-assistant/1.0"},
    )
    response.raise_for_status()
    return {
        "title": title,
        "url": url,
        "sections": extract_page_text(response.text),
    }


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "source"


def scrape_all(output_dir: Path) -> list[Path]:
    """Scrape every configured OWASP source and persist one JSON file per source."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for title, url in OWASP_SOURCES.items():
        payload = scrape_source(title, url)
        path = output_dir / f"{_slugify(title)}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(path)

    return written


if __name__ == "__main__":
    paths = scrape_all(Path("data/raw"))
    print(f"Scraped {len(paths)} OWASP sources")
    for path in paths:
        print(path)

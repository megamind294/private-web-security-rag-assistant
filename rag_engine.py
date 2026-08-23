from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()


class RAGEngine:
    def __init__(self, index_path: Path):
        payload = joblib.load(Path(index_path))
        self.vectorizer = payload["vectorizer"]
        self.matrix = payload["matrix"]
        self.chunks = payload["chunks"]

    def retrieve(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if top_k <= 0:
            return []

        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk = dict(self.chunks[int(index)])
            chunk["score"] = float(scores[int(index)])
            results.append(chunk)
        return results

    @staticmethod
    def _citations(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "chunk_id": item["chunk_id"],
                "score": float(item["score"]),
            }
            for item in retrieved
        ]

    def _extractive_answer(self, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        citations = self._citations(retrieved)
        if not retrieved or max(item["score"] for item in retrieved) <= 0:
            answer = (
                "The local OWASP knowledge base did not find a confident match for this question. "
                "Try asking about CSRF, XSS, SQL injection, authentication, authorization, or session management."
            )
        else:
            best = [item["text"].strip() for item in retrieved[:2] if item["text"].strip()]
            answer = "\n\n".join(best)

        return {
            "question": question,
            "mode": "extractive",
            "answer": answer,
            "citations": citations,
        }

    @staticmethod
    def _context_text(retrieved: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{item['chunk_id']}] {item['source_title']}\n{item['text']}"
            for item in retrieved
        )

    def _answer_with_openai(self, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
        system = (
            "You are a web-security assistant. Answer only from the supplied OWASP context. "
            "Do not invent facts or citations. If the context is insufficient, say so."
        )
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"OWASP context:\n{self._context_text(retrieved)}\n\nQuestion: {question}",
                    },
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("output_text")
        if not answer:
            for item in payload.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        answer = content["text"]
                        break
                if answer:
                    break
        if not answer:
            raise RuntimeError("OpenAI response did not contain output text")
        return {
            "question": question,
            "mode": "openai",
            "answer": answer,
            "citations": self._citations(retrieved),
        }

    def _answer_with_ollama(self, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not base_url or not model:
            raise RuntimeError("Ollama is not configured")

        prompt = (
            "Answer only from the supplied OWASP context. Do not invent citations. "
            "If context is insufficient, say so.\n\n"
            f"OWASP context:\n{self._context_text(retrieved)}\n\nQuestion: {question}"
        )
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=10,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama response did not contain text")
        return {
            "question": question,
            "mode": "ollama",
            "answer": answer,
            "citations": self._citations(retrieved),
        }

    def answer(self, question: str, mode: str = "security") -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        retrieved = self.retrieve(question, top_k=4)
        mode = mode.strip().lower() or "security"

        if mode == "extractive":
            return self._extractive_answer(question, retrieved)

        if mode == "local":
            try:
                return self._answer_with_ollama(question, retrieved)
            except (requests.RequestException, RuntimeError, ValueError):
                return self._extractive_answer(question, retrieved)

        if mode != "security":
            raise ValueError("Unsupported mode")

        if os.getenv("OPENAI_API_KEY", "").strip():
            try:
                return self._answer_with_openai(question, retrieved)
            except (requests.RequestException, RuntimeError, ValueError):
                pass

        if os.getenv("OLLAMA_BASE_URL", "").strip() and os.getenv("OLLAMA_MODEL", "").strip():
            try:
                return self._answer_with_ollama(question, retrieved)
            except (requests.RequestException, RuntimeError, ValueError):
                pass

        return self._extractive_answer(question, retrieved)

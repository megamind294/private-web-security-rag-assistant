from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from rag_engine import RAGEngine


DEFAULT_INDEX_PATH = Path("data/index/rag_index.pkl")


def create_app(engine: Any = None) -> Flask:
    app = Flask(__name__)

    rag_engine = engine
    if rag_engine is None and DEFAULT_INDEX_PATH.exists():
        rag_engine = RAGEngine(DEFAULT_INDEX_PATH)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "engine_ready": rag_engine is not None,
                "index_path": str(DEFAULT_INDEX_PATH),
            }
        )

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        mode = str(payload.get("mode", "security")).strip().lower() or "security"

        if not question:
            return jsonify({"error": "Question must not be empty"}), 400
        if rag_engine is None:
            return (
                jsonify(
                    {
                        "error": (
                            "RAG index is not ready. Run the OWASP scraper and index builder first."
                        )
                    }
                ),
                503,
            )

        try:
            return jsonify(rag_engine.answer(question, mode=mode))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

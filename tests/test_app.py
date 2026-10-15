import pytest

from app import create_app


class FakeEngine:
    def answer(self, question: str, mode: str = "security"):
        return {
            "question": question,
            "mode": "extractive",
            "answer": "Use parameterized queries.",
            "citations": [],
        }


@pytest.fixture
def client():
    app = create_app(engine=FakeEngine())
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Private Web Security Assistant" in response.data
    assert b"Ask a security question" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ask_rejects_blank_question(client):
    response = client.post("/api/ask", json={"question": ""})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Question must not be empty"


def test_ask_returns_engine_payload(client):
    response = client.post("/api/ask", json={"question": "How do I prevent SQL injection?"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["answer"] == "Use parameterized queries."
    assert payload["mode"] == "extractive"

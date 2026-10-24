from fastapi.testclient import TestClient
import app as app_module
from support_rag import RAGBackendError, RAGService


class FakeDoc:
    page_content = "Cashless health insurance is a type of plan where the insurer settles the bill."
    metadata = {"file_name": "faq.docx"}


class FakeRAG:
    def initialize(self):
        return 1

    def answer_question(self, question):
        if "backend-failure" in question:
            raise RAGBackendError("simulated failure")
        if "unsupported" in question.lower():
            return "The answer is not available in the provided documents.", [FakeDoc()], 0.91, False
        return "The insurer settles the bill directly. Source: faq.docx", [FakeDoc()], 0.12, True


def test_health(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_valid_question(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.post("/ask", json={"question": "What is cashless hospitalization?"})
        assert response.status_code == 200
        assert response.json()["grounded"] is True
        assert response.json()["sources"][0]["file_name"] == "faq.docx"
        assert response.json()["latency_ms"] >= 0


def test_empty_question_rejected(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 422


def test_long_question_rejected(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.post("/ask", json={"question": "x" * 1001})
        assert response.status_code == 422


def test_backend_failure_is_controlled(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.post("/ask", json={"question": "backend-failure"})
        assert response.status_code == 502
        assert "temporarily unavailable" in response.json()["detail"]
        assert "traceback" not in response.text.lower()


def test_unsupported_question_returns_fallback(monkeypatch):
    monkeypatch.setattr(app_module, "RAGService", FakeRAG)
    with TestClient(app_module.app) as client:
        response = client.post("/ask", json={"question": "unsupported topic"})
        assert response.status_code == 200
        assert response.json()["grounded"] is False
        assert "not available" in response.json()["answer"].lower()


def test_ollama_provider_uses_local_generation(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "Local model answer"}

    calls = {}

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.setattr("support_rag.httpx.post", fake_post)

    service = RAGService()
    answer = service._generate_text("test prompt")

    assert answer == "Local model answer"
    assert calls["url"] == "http://127.0.0.1:11434/api/generate"
    assert calls["kwargs"]["json"]["model"] == "test-model"
    assert calls["kwargs"]["json"]["stream"] is False


def test_invalid_llm_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    service = RAGService()

    try:
        service._generate_text("test prompt")
    except ValueError as exc:
        assert "LLM_PROVIDER" in str(exc)
    else:
        raise AssertionError("Unsupported provider should raise ValueError")

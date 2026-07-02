from fastapi.testclient import TestClient

import app.api as api_module
from tests.conftest import make_agent


def test_health_returns_ok():
    with TestClient(api_module.app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_schema_compliance(shared_retriever, monkeypatch):
    with TestClient(api_module.app) as client:
        api_module._agent = make_agent(
            shared_retriever,
            extractions=[{"role_title": "Java developer", "skills": ["Java"], "seniority": "mid-level"}],
        )
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "Hiring a Java developer"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert isinstance(body["reply"], str)
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["end_of_conversation"], bool)
    for rec in body["recommendations"]:
        assert set(rec.keys()) == {"name", "url", "test_type"}


def test_chat_never_returns_500_even_if_agent_throws(shared_retriever, monkeypatch):
    with TestClient(api_module.app) as client:
        agent = make_agent(shared_retriever)

        def boom(messages):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(agent, "handle_chat", boom)
        api_module._agent = agent
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "anything"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert isinstance(body["reply"], str) and body["reply"]


def test_chat_rejects_malformed_request_body():
    with TestClient(api_module.app) as client:
        resp = client.post("/chat", json={"messages": [{"role": "not-a-valid-role", "content": "hi"}]})
    assert resp.status_code == 422


def test_chat_empty_messages_list_does_not_crash():
    with TestClient(api_module.app) as client:
        resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["reply"], str) and body["reply"]

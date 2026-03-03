import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_no_docs():
    response = client.post("/api/query", json={"session_id": "test_session", "query": "hello"})
    assert response.status_code == 400
    assert "No documents uploaded" in response.json()["detail"]

def test_common_sections_no_docs():
    response = client.post("/api/common-sections", json={"session_id": "test_session", "query": ""})
    assert response.status_code == 400
    assert "No documents uploaded" in response.json()["detail"]

def test_list_documents_empty():
    response = client.get("/api/documents/list/empty_session")
    assert response.status_code == 200
    assert response.json()["documents"] == []

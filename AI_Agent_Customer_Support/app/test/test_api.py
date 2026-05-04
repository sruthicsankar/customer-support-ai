import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_login():
    """Test login with correct credentials."""
    response = client.post("/login", json={"username": "user", "password": "pass"})
    assert response.status_code == 200
    assert response.json()["message"] == "Login successful"
    return "user"

def test_chat():
    """Test chat with logged-in user."""
    username = test_login()
    response = client.post(
        "/chat",
        json={"query": "What is the return policy?", "username": username}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_history():
    """Test history retrieval for logged-in user."""
    username = test_login()
    client.post("/chat", json={"query": "Test query", "username": username})
    response = client.post("/history", json={"username": username})
    assert response.status_code == 200
    assert "history" in response.json()
    assert len(response.json()["history"]) >= 2

def test_logout():
    """Test logout for logged-in user."""
    username = test_login()
    response = client.post("/logout", json={"username": username})
    assert response.status_code == 200
    assert response.json()["message"] == "Logout successful"
    # Verify user must log in again
    response = client.post("/chat", json={"query": "Test", "username": username})
    assert response.status_code == 401
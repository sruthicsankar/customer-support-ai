import chromadb
from typing import List, Dict
from app.logger import logger

class HistoryManager:
    def __init__(self):
        # Initialize ChromaDB for chat history
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("chat_history")

    def get_history(self, username: str) -> List[Dict]:
        """Retrieve chat history for a user."""
        results = self.collection.get(where={"username": username})
        return [
            {"role": meta["role"], "content": doc}
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", []))
        ]

    def add_message(self, username: str, message: Dict):
        """Add a message to chat history."""
        self.collection.add(
            documents=[message["content"]],
            metadatas=[{"username": username, "role": message["role"]}],
            ids=[f"{username}_{len(self.get_history(username)) + 1}"]
        )
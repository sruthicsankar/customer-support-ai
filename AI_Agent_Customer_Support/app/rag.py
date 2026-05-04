import chromadb
import requests
from typing import List, Dict
from app.config import Config
from app.logger import logger

class RAGPipeline:
    def __init__(self):
        # Hard-coded support documents
        self.documents = [
            "Document 1: Title: 'Return Policy' Content: Returns are accepted within 30 days of purchase with a valid receipt. Refunds are processed in 5-7 business days. Contact support@universal.com for assistance.",
            "Document 2: Title: 'Shipping Information' Content: We offer free standard shipping on orders over $50. Delivery takes 3-5 business days. Express shipping costs $10 and delivers in 1-2 days.",
            "Document 3: Title: 'Account Management' Content: To reset your password, click 'Forgot Password' on the login page. Account updates can be made in the profile section of our website."
        ]
        # Initialize ChromaDB for documents
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("support_docs")
        self.use_embeddings = True
        # Generate embeddings for documents
        embeddings = self.get_embeddings(self.documents)
        if embeddings and all(embeddings):  # Check if embeddings are valid
            try:
                self.collection.add(
                    documents=self.documents,
                    embeddings=embeddings,
                    ids=[f"doc_{i}" for i in range(len(self.documents))]
                )
                logger.info("Documents added to ChromaDB with embeddings")
            except Exception as e:
                logger.error(f"ChromaDB add failed: {str(e)}")
                self.use_embeddings = False
        else:
            logger.warning("Embeddings unavailable, using keyword-based retrieval")
            self.use_embeddings = False

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the embeddings API."""
        payload = {"input": texts}
        headers = {"Authorization": f"Bearer {Config.API_KEY}", "Content-Type": "application/json"}
        try:
            response = requests.post(Config.EMBED_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json().get("data", [])
            logger.info(f"Embeddings API response: {response.text}")
            print(f"Embeddings API response: {response.text}")  # Debug to console
            if not data or not isinstance(data, list) or "embeddings" not in data[0]:
                logger.error("Invalid embeddings response format")
                return [[] for _ in texts]
            return data[0].get("embeddings", [[] for _ in texts])
        except Exception as e:
            logger.error(f"Embedding API failed: {str(e)}")
            print(f"Embedding API error: {str(e)}")  # Debug to console
            return [[] for _ in texts]

    def hybrid_retrieval(self, query: str) -> List[Dict]:
        """Retrieve documents using embeddings or fallback to keywords."""
        logger.info(f"Retrieving documents for query: {query}")
        if self.use_embeddings:
            query_embedding = self.get_embeddings([query])[0]
            if query_embedding:
                try:
                    results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
                    documents = [
                        {"content": doc, "id": id}
                        for doc, id in zip(results.get("documents", [[]])[0], results.get("ids", [[]])[0])
                    ]
                    logger.info(f"Retrieved {len(documents)} documents from ChromaDB")
                    return self.rerank(query, documents)
                except Exception as e:
                    logger.error(f"ChromaDB query failed: {str(e)}")
                    print(f"ChromaDB query error: {str(e)}")  # Debug to console
                    self.use_embeddings = False
        # Fallback to keyword-based retrieval
        logger.info("Using keyword-based retrieval")
        keywords = query.lower().split()
        results = []
        for i, doc in enumerate(self.documents):
            score = sum(1 for kw in keywords if kw in doc.lower())
            if score > 0:
                results.append({"content": doc, "id": f"doc_{i}", "score": score})
        ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:3]
        logger.info(f"Keyword retrieval returned {len(ranked)} documents")
        return ranked

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using the reranker API."""
        if not documents:
            logger.info("No documents to rerank")
            return []
        payload = {
            "query": query,
            "documents": [doc["content"] for doc in documents]
        }
        headers = {"Authorization": f"Bearer {Config.API_KEY}", "Content-Type": "application/json"}
        try:
            response = requests.post(Config.RERANK_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            scores = response.json().get("scores", [0] * len(documents))
            logger.info(f"Reranker API response: {response.text}")
            print(f"Reranker API response: {response.text}")  # Debug to console
            ranked = [
                {"content": doc["content"], "id": doc["id"], "score": score}
                for doc, score in zip(documents, scores)
            ]
            return sorted(ranked, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"Reranker API failed: {str(e)}")
            print(f"Reranker API error: {str(e)}")  # Debug to console
            return documents

    def compress_context(self, documents: List[Dict]) -> str:
        """Combine document content for context."""
        if not documents:
            logger.info("No documents for context")
            return ""
        context = " ".join(doc["content"] for doc in documents)
        logger.info(f"Compressed context: {context[:50]}...")
        return context

    def generate_response(self, query: str, context: str, history: List[Dict]) -> str:
        """Generate a response using context and history."""
        logger.info(f"Generating response for query: {query}")
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        prompt = f"History:\n{history_text}\n\nContext:\n{context or 'No info available'}\n\nQuery:\n{query}\n\nAnswer as a support agent:"
        payload = {
            "model": "usf-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 200
        }
        headers = {"Authorization": f"Bearer {Config.API_KEY}", "Content-Type": "application/json"}
        try:
            response = requests.post(Config.LLM_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Chat API response: {response.text}")
            print(f"Chat API response: {response.text}")  # Debug to console
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Chat API failed: {str(e)}")
            print(f"Chat API error: {str(e)}")  # Debug to console
            return "Sorry, I couldn't process your request."
from dotenv import load_dotenv
import os

load_dotenv()
class Config:
    API_KEY = os.getenv("API_KEY")
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://api.us.inc/usf/v1/hiring/chat/completions")
    EMBED_ENDPOINT = os.getenv("EMBED_ENDPOINT", "https://api.us.inc/usf/v1/embed/embeddings")
    RERANK_ENDPOINT = os.getenv("RERANK_ENDPOINT", "https://api.us.inc/usf/v1/embed/reranker")
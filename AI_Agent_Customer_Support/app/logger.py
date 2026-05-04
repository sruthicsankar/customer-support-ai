import logging

# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger("Chatbot")
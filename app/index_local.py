from pathlib import Path
import chromadb
import os
from bs4 import BeautifulSoup
from loguru import logger
from dotenv import load_dotenv

# Load .env file
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", str(Path(__file__).parent / "data"))
CLIENT_ID = "rsms"
DOCS_PATH = Path(BASE_DIR) / CLIENT_ID / "documents"
COLLECTION_NAME = f"{CLIENT_ID}_documents"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fix INDEX_PATH - remove extra spaces
INDEX_PATH = f"{BASE_DIR}/{CLIENT_ID}/index_store"

# Create directory if it doesn't exist
Path(INDEX_PATH).mkdir(parents=True, exist_ok=True)

# Initialize OpenAI client if key is available
if OPENAI_API_KEY:
    logger.info("OPENAI_API_KEY loaded, embeddings enabled.")
    import openai
    openai.api_key = OPENAI_API_KEY
else:
    logger.error("OPENAI_API_KEY not set in .env. Indexing will proceed without embeddings if ChromaDB handles it.")

# Initialize ChromaDB
client = chromadb.PersistentClient(path=INDEX_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Index documents
logger.info(f"Indexing documents from {DOCS_PATH}")
for file_path in DOCS_PATH.glob("*.html"):
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
        if text.strip():
            try:
                if OPENAI_API_KEY:
                    response = openai.embeddings.create(input=[text], model="text-embedding-ada-002")
                    embedding = response.data[0].embedding
                    collection.add(
                        embeddings=[embedding],
                        documents=[text],
                        metadatas=[{"filename": file_path.name}],
                        ids=[file_path.name]
                    )
                else:
                    collection.add(
                        documents=[text],
                        metadatas=[{"filename": file_path.name}],
                        ids=[file_path.name]
                    )
                logger.info(f"Indexed {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to index {file_path.name}: {e}")
        else:
            logger.warning(f"No text extracted from {file_path.name}")

logger.info(f"Total documents indexed: {collection.count()}")
print("Collection peek:", collection.peek())
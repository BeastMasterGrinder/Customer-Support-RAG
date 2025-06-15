from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_langchain_db"

# Data files
PRODUCT_DOCS_FILE = DATA_DIR / "product_docs.json"
SUPPORT_TICKETS_FILE = DATA_DIR / "support_tickets.json"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store settings
COLLECTION_NAME = "cloudsync_support"

# Embedding settings
EMBEDDING_MODEL = "models/embedding-001"  # Google's embedding model

# Document types
DOC_TYPES = {
    "PRODUCT_DOC": "product_doc",
    "SUPPORT_TICKET": "support_ticket"
} 
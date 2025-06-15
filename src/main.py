import logging
from pathlib import Path
from data_processing.document_loader import CloudSyncDocumentLoader
from data_processing.intelligent_splitter import IntelligentSplitter
from embeddings.embedding_manager import EmbeddingManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting document processing pipeline")
        
        # 1. Load documents
        logger.info("Loading documents")
        document_loader = CloudSyncDocumentLoader()
        documents = document_loader.load_all_documents()
        logger.info(f"Loaded {len(documents)} documents")
        
        # 2. Split documents
        logger.info("Splitting documents with intelligent chunking")
        splitter = IntelligentSplitter()
        split_documents = splitter.split_documents(documents)
        logger.info(f"Created {len(split_documents)} chunks")
        
        # 3. Initialize embedding manager and add documents
        logger.info("Initializing embedding manager")
        embedding_manager = EmbeddingManager()
        
        logger.info("Adding documents to vector store")
        embedding_manager.add_documents(split_documents)
        
        # 4. Get and log collection statistics
        stats = embedding_manager.get_collection_stats()
        logger.info(f"Vector store statistics: {stats}")
        
        logger.info("Document processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
import logging
from pathlib import Path
from data_processing.document_loader import CloudSyncDocumentLoader
from data_processing.intelligent_splitter import IntelligentSplitter
from embeddings.embedding_manager import EmbeddingManager
from embeddings.smart_retrieval import SmartRetrieval
from answer_generation import AnswerGenerator, AnswerFormatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_smart_retrieval(retrieval: SmartRetrieval, answer_gen: AnswerGenerator, formatter: AnswerFormatter):
    """Test the smart retrieval system with various queries."""
    test_cases = [
        # Basic queries
        {
            "query": "How do I login to the system?",
            "user_version": None
        },
        {
            "query": "Why isn't my data syncing?",
            "user_version": None
        },
        # Version-specific queries
        {
            "query": "How to use the new dashboard in version 2.1?",
            "user_version": "2.0"  # User asking about features in a newer version
        },
        {
            "query": "Configure authentication settings",
            "user_version": "1.5"  # User on older version
        },
        {
            "query": "How to enable real-time sync in v2.0?",
            "user_version": None  # Version mentioned in query
        },
        # Migration scenarios
        {
            "query": "What's new in version 2.1?",
            "user_version": "2.0"
        },
        {
            "query": "How to migrate from v1.5 to v2.0?",
            "user_version": None
        }
    ]
    
    logger.info("Testing smart retrieval and answer generation system...")
    for test_case in test_cases:
        query = test_case["query"]
        user_version = test_case["user_version"]
        
        logger.info(f"\nQuery: {query}")
        if user_version:
            logger.info(f"User Version: {user_version}")
        
        # Get relevant documents
        retrieved_docs = retrieval.search(query)
        
        # Generate and format answer
        generated_answer = answer_gen.generate_answer(query, retrieved_docs, user_version)
        formatted_answer = formatter.format_answer(generated_answer)
        
        logger.info("Generated Answer:")
        logger.info(formatted_answer)
        logger.info("-" * 80)

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
        
        # 4. Initialize smart retrieval
        logger.info("Initializing smart retrieval system")
        smart_retrieval = SmartRetrieval(embedding_manager)
        
        # 5. Initialize answer generation components
        logger.info("Initializing answer generation system")
        answer_generator = AnswerGenerator()
        answer_formatter = AnswerFormatter()
        
        # 6. Test the complete system
        test_smart_retrieval(smart_retrieval, answer_generator, answer_formatter)
        
        # 7. Get and log collection statistics
        stats = embedding_manager.get_collection_stats()
        logger.info(f"Vector store statistics: {stats}")
        
        logger.info("Document processing pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
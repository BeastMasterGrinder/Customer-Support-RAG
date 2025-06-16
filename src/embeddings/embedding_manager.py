from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from config.config import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR
import os
import socket
import time
from google.api_core import retry
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages document embeddings and vector store operations.
    """
    
    def __init__(self):
        # Disable IPv6 to avoid connection issues
        # This forces the use of IPv4 for all connections
        socket.setdefaulttimeout(120)  # Increase socket timeout to 120 seconds
        if hasattr(socket, 'AF_INET6'):
            try:
                socket.socket(socket.AF_INET6, socket.SOCK_STREAM).close()
                # Force IPv4
                os.environ['GRPC_DNS_RESOLVER'] = 'native'
                os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
            except socket.error:
                pass

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it using: export GOOGLE_API_KEY='your-api-key'"
            )
        
        # Configure custom retry strategy
        custom_retry = retry.Retry(
            initial=1.0,  # Initial delay in seconds
            maximum=60.0,  # Maximum delay between retries
            multiplier=2.0,  # Multiply delay by this factor after each retry
            predicate=retry.if_exception_type(
                ConnectionError,
                TimeoutError,
                socket.timeout,
                socket.error
            ),
            deadline=300.0  # Total timeout for all retries
        )
            
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
            request_timeout=120,  # Increase request timeout to 120 seconds
            retry=custom_retry
        )
        
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR)
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store with improved error handling and retries.
        """
        # Filter complex metadata from documents
        filtered_documents = []
        for doc in documents:
            # Convert list values to comma-separated strings in metadata
            filtered_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    filtered_metadata[key] = ','.join(str(v) for v in value)
                else:
                    filtered_metadata[key] = value
                    
            # Create new document with filtered metadata
            filtered_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=filtered_metadata
                )
            )
        
        # Add documents in smaller batches to avoid timeouts
        batch_size = 5
        total_batches = (len(filtered_documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(filtered_documents), batch_size):
            batch = filtered_documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Processing batch {batch_num}/{total_batches} (Documents {i+1}-{min(i+batch_size, len(filtered_documents))})")
                    self.vector_store.add_documents(batch)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Error processing batch {batch_num}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to process batch {batch_num} after {max_retries} attempts")
                        raise
    
    def similarity_search(self, query: str, k: int = 4, filter_dict: dict = None) -> List[Document]:
        """
        Search for similar documents with improved error handling.
        """
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                return self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Search failed, attempt {attempt + 1}/{max_retries}: {str(e)}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Search failed after all retry attempts")
                    raise
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store collection.
        """
        collection = self.vector_store._collection
        return {
            "total_documents": collection.count(),
            "collection_name": COLLECTION_NAME
        } 
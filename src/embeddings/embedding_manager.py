from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from config.config import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR
import os

class EmbeddingManager:
    """
    Manages document embeddings and vector store operations.
    """
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it using: export GOOGLE_API_KEY='your-api-key'"
            )
            
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key  # Using google_api_key parameter instead of api_key
        )
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR)
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        This method filters out complex metadata types that ChromaDB doesn't support
        before adding the documents to the vector store.
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
            
        # Add documents to ChromaDB
        self.vector_store.add_documents(filtered_documents)
    
    def similarity_search(self, query: str, k: int = 4, filter_dict: dict = None) -> List[Document]:
        """
        Search for similar documents.
        """
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store collection.
        """
        collection = self.vector_store._collection
        return {
            "total_documents": collection.count(),
            "collection_name": COLLECTION_NAME
        } 
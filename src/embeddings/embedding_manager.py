from typing import List
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from config.config import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR
import os
class EmbeddingManager:
    """
    Manages document embeddings and vector store operations.
    """
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR)
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        """
        # Add documents to ChromaDB
        self.vector_store.add_documents(documents)
        # Persist the database
        self.vector_store.persist()
    
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
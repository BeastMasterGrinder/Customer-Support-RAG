import json
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from config.config import DOC_TYPES, PRODUCT_DOCS_FILE, SUPPORT_TICKETS_FILE

class CloudSyncDocumentLoader:
    """Loader for CloudSync documentation and support tickets."""
    
    @staticmethod
    def load_product_docs() -> List[Document]:
        """Load product documentation from JSON file."""
        with open(PRODUCT_DOCS_FILE, 'r') as f:
            data = json.load(f)
            docs = []
            for doc in data['product_docs']:
                docs.append(
                    Document(
                        page_content=doc['content'],
                        metadata={
                            'source': DOC_TYPES['PRODUCT_DOC'],
                            'id': doc['id'],
                            'title': doc['title'],
                            'type': doc['type'],
                            'version': doc['version'],
                            'tags': doc['tags'],
                            'last_updated': doc.get('last_updated')
                        }
                    )
                )
            return docs
    
    @staticmethod
    def load_support_tickets() -> List[Document]:
        """Load support tickets from JSON file."""
        with open(SUPPORT_TICKETS_FILE, 'r') as f:
            data = json.load(f)
            docs = []
            for ticket in data['support_tickets']:
                docs.append(
                    Document(
                        page_content=ticket['content'],
                        metadata={
                            'source': DOC_TYPES['SUPPORT_TICKET'],
                            'id': ticket['id'],
                            'title': ticket['title'],
                            'status': ticket['status'],
                            'category': ticket['category'],
                            'priority': ticket['priority'],
                            'user_version': ticket['user_version'],
                            'created_date': ticket['created_date'],
                            'resolved_date': ticket['resolved_date'],
                            'tags': ticket['tags']
                        }
                    )
                )
            return docs
    
    @classmethod
    def load_all_documents(cls) -> List[Document]:
        """Load all documents from both sources."""
        product_docs = cls.load_product_docs()
        support_tickets = cls.load_support_tickets()
        return product_docs + support_tickets 
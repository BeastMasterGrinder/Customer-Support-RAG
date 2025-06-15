from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.text_processing import (
    extract_numbered_lists,
    identify_section_boundaries,
    find_step_sequences,
    merge_overlapping_regions
)
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

class IntelligentSplitter:
    """
    Intelligent document splitter that preserves document structure and context.
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _identify_preserve_regions(self, text: str) -> List[dict]:
        """
        Identify regions in text that should be preserved together.
        """
        # Find all special regions
        numbered_lists = extract_numbered_lists(text)
        sections = identify_section_boundaries(text)
        step_sequences = find_step_sequences(text)
        
        # Combine all regions
        all_regions = numbered_lists + sections + step_sequences
        
        # Merge overlapping regions
        return merge_overlapping_regions(all_regions)
    
    def _split_with_preserved_regions(self, text: str, preserve_regions: List[dict]) -> List[str]:
        """
        Split text while keeping preserved regions intact.
        """
        chunks = []
        last_end = 0
        
        for region in preserve_regions:
            # Split text before region if it exists
            if last_end < region['start']:
                pre_text = text[last_end:region['start']]
                if pre_text.strip():
                    chunks.extend(self.base_splitter.split_text(pre_text))
            
            # Add the preserved region as a whole
            region_text = region['content']
            if len(region_text) > self.chunk_size * 2:
                # If region is too large, split it with larger chunk size
                temp_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size * 2,
                    chunk_overlap=self.chunk_overlap * 2
                )
                chunks.extend(temp_splitter.split_text(region_text))
            else:
                chunks.append(region_text)
            
            last_end = region['end']
        
        # Split any remaining text
        if last_end < len(text):
            remaining_text = text[last_end:]
            if remaining_text.strip():
                chunks.extend(self.base_splitter.split_text(remaining_text))
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents while preserving important structure and metadata.
        """
        split_docs = []
        
        for doc in documents:
            # Identify regions to preserve
            preserve_regions = self._identify_preserve_regions(doc.page_content)
            
            # Split the document
            chunks = self._split_with_preserved_regions(doc.page_content, preserve_regions)
            
            # Create new documents for each chunk
            for i, chunk in enumerate(chunks):
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    )
                )
        
        return split_docs 
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import re
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import os

@dataclass
class Citation:
    doc_id: str
    doc_title: str
    doc_type: str
    section: Optional[str]
    confidence: float
    version: Optional[str]
    last_updated: Optional[str]

@dataclass
class VersionInfo:
    current_version: str
    available_versions: List[str]
    is_latest: bool
    next_version: Optional[str] = None
    migration_info: Optional[str] = None

@dataclass
class GeneratedAnswer:
    answer_text: str
    citations: List[Citation]
    confidence_score: float
    has_insufficient_info: bool
    has_conflicting_info: bool
    has_outdated_info: bool
    outdated_versions: List[str]
    version_info: Optional[VersionInfo] = None
    is_version_specific: bool = False

class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key
        )
    
    def _extract_section(self, doc: Document) -> Optional[str]:
        """Extract section information from document content."""
        content = doc.page_content
        # Look for markdown-style headers
        headers = [line for line in content.split('\n') if line.startswith('#')]
        if headers:
            return headers[0].lstrip('#').strip()
        return None
    
    def _extract_version_from_query(self, query: str) -> Optional[str]:
        """Extract version information from the query."""
        # Match common version patterns
        patterns = [
            r'v\d+\.\d+(?:\.\d+)?',  # v1.0, v2.1.3
            r'version\s+\d+\.\d+(?:\.\d+)?',  # version 1.0
            r'\d+\.\d+(?:\.\d+)?\s+version'   # 1.0 version
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                version = match.group().lower().replace('v', '').replace('version', '').strip()
                return version
        return None
    
    def _get_version_info(self, docs: List[Tuple[Document, float]], query_version: Optional[str] = None) -> VersionInfo:
        """Get version information from available documents."""
        versions = set()
        for doc, _ in docs:
            version = doc.metadata.get('version') or doc.metadata.get('user_version')
            if version:
                versions.add(version)
        
        if not versions:
            return None
        
        sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split('.')])
        latest_version = sorted_versions[-1]
        
        if query_version:
            is_latest = query_version == latest_version
            next_version = None
            try:
                version_idx = sorted_versions.index(query_version)
                if version_idx < len(sorted_versions) - 1:
                    next_version = sorted_versions[version_idx + 1]
            except ValueError:
                pass
            
            return VersionInfo(
                current_version=query_version,
                available_versions=sorted_versions,
                is_latest=is_latest,
                next_version=next_version
            )
        
        return VersionInfo(
            current_version=latest_version,
            available_versions=sorted_versions,
            is_latest=True
        )
    
    def _filter_version_specific_docs(self, docs: List[Tuple[Document, float]], version: str) -> List[Tuple[Document, float]]:
        """Filter documents to match specific version."""
        filtered_docs = []
        for doc, score in docs:
            doc_version = doc.metadata.get('version') or doc.metadata.get('user_version')
            if doc_version == version:
                filtered_docs.append((doc, score * 1.2))  # Boost score for version-specific matches
        return filtered_docs if filtered_docs else docs  # Fall back to all docs if no version matches
    
    def _get_migration_info(self, docs: List[Tuple[Document, float]], from_version: str, to_version: str) -> Optional[str]:
        """Extract migration information between versions."""
        migration_docs = []
        for doc, _ in docs:
            content = doc.page_content.lower()
            if ('migration' in content or 'upgrade' in content) and \
               (from_version in content and to_version in content):
                migration_docs.append(doc)
        
        if migration_docs:
            return "\n".join([doc.page_content for doc in migration_docs])
        return None
    
    def _check_version_conflicts(self, docs: List[Tuple[Document, float]]) -> List[str]:
        """Identify different versions present in retrieved documents."""
        versions = set()
        for doc, _ in docs:
            version = doc.metadata.get('version') or doc.metadata.get('user_version')
            if version:
                versions.add(version)
        return list(versions) if len(versions) > 1 else []
    
    def _format_technical_steps(self, text: str) -> str:
        """Format technical instructions with proper markdown."""
        # Convert "Step X:" pattern to numbered list
        text = text.replace("Step 1:", "1.")
        text = text.replace("Step 2:", "2.")
        text = text.replace("Step 3:", "3.")
        
        # Add newlines before lists for proper markdown rendering
        text = text.replace("\n•", "\n\n•")
        text = text.replace("\n1.", "\n\n1.")
        
        return text
    
    def _create_citation(self, doc: Document, score: float) -> Citation:
        """Create a citation object from a document."""
        return Citation(
            doc_id=doc.metadata.get('id', 'unknown'),
            doc_title=doc.metadata.get('title', 'Untitled'),
            doc_type=doc.metadata.get('source', 'unknown'),
            section=self._extract_section(doc),
            confidence=score,
            version=doc.metadata.get('version') or doc.metadata.get('user_version'),
            last_updated=doc.metadata.get('last_updated') or doc.metadata.get('resolved_date')
        )
    
    def _has_conflicting_information(self, docs: List[Tuple[Document, float]]) -> bool:
        """Check if there are conflicting pieces of information."""
        seen_content = set()
        for doc, _ in docs:
            content_key = doc.page_content.lower()
            if "not" in content_key and content_key.replace("not", "").strip() in seen_content:
                return True
            seen_content.add(content_key)
        return False
    
    def generate_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]], user_version: Optional[str] = None) -> GeneratedAnswer:
        """Generate a comprehensive answer with citations and handle edge cases."""
        if not retrieved_docs:
            return GeneratedAnswer(
                answer_text="I don't have enough information to answer this question.",
                citations=[],
                confidence_score=0.0,
                has_insufficient_info=True,
                has_conflicting_info=False,
                has_outdated_info=False,
                outdated_versions=[]
            )
        
        # Extract version from query if not provided
        query_version = user_version or self._extract_version_from_query(query)
        version_info = self._get_version_info(retrieved_docs, query_version)
        
        # Filter and prioritize version-specific documents
        if query_version:
            retrieved_docs = self._filter_version_specific_docs(retrieved_docs, query_version)
            
            # Get migration info if asking about newer version features
            if version_info and version_info.next_version:
                migration_info = self._get_migration_info(
                    retrieved_docs,
                    version_info.current_version,
                    version_info.next_version
                )
                if migration_info:
                    version_info.migration_info = migration_info
        
        # Create citations
        citations = [self._create_citation(doc, score) for doc, score in retrieved_docs]
        
        # Check for edge cases
        outdated_versions = self._check_version_conflicts(retrieved_docs)
        has_conflicting = self._has_conflicting_information(retrieved_docs)
        
        # Prepare context for LLM
        context_parts = [
            f"Document {i+1} ({doc.metadata.get('source')}): {doc.page_content}"
            for i, (doc, _) in enumerate(retrieved_docs)
        ]
        
        if version_info and version_info.migration_info:
            context_parts.append(f"Migration Information: {version_info.migration_info}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer the question: {query}

Context:
{context}

Requirements:
1. If information is insufficient, start with "I don't have enough information about..."
2. If there are conflicting pieces of information, acknowledge them explicitly
3. If there are version differences, mention them clearly
4. Format any technical steps using numbered lists or bullet points
5. Be concise but comprehensive
6. Do not make up information not present in the context
7. If the query is about a specific version, focus on that version's information
8. If features are only available in newer versions, mention the version requirements

Answer:"""

        response = self.llm.invoke(prompt)
        answer_text = self._format_technical_steps(response.content)
        
        # Calculate overall confidence
        confidence_score = sum(score for _, score in retrieved_docs) / len(retrieved_docs)
        
        return GeneratedAnswer(
            answer_text=answer_text,
            citations=citations,
            confidence_score=confidence_score,
            has_insufficient_info=confidence_score < 0.3,
            has_conflicting_info=has_conflicting,
            has_outdated_info=bool(outdated_versions),
            outdated_versions=outdated_versions,
            version_info=version_info,
            is_version_specific=bool(query_version)
        ) 
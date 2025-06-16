from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
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
class GeneratedAnswer:
    answer_text: str
    citations: List[Citation]
    confidence_score: float
    has_insufficient_info: bool
    has_conflicting_info: bool
    has_outdated_info: bool
    outdated_versions: List[str]

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
        # Compare document contents for potential conflicts
        seen_content = set()
        for doc, _ in docs:
            content_key = doc.page_content.lower()
            if "not" in content_key and content_key.replace("not", "").strip() in seen_content:
                return True
            seen_content.add(content_key)
        return False
    
    def generate_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> GeneratedAnswer:
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
        
        # Create citations
        citations = [self._create_citation(doc, score) for doc, score in retrieved_docs]
        
        # Check for edge cases
        outdated_versions = self._check_version_conflicts(retrieved_docs)
        has_conflicting = self._has_conflicting_information(retrieved_docs)
        
        # Prepare context for LLM
        context = "\n\n".join([
            f"Document {i+1} ({doc.metadata.get('source')}): {doc.page_content}"
            for i, (doc, _) in enumerate(retrieved_docs)
        ])
        
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
            outdated_versions=outdated_versions
        ) 
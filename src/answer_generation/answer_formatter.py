from typing import List
from datetime import datetime
from .answer_generator import GeneratedAnswer, Citation

class AnswerFormatter:
    @staticmethod
    def _format_confidence_indicator(confidence: float) -> str:
        """Convert confidence score to human-readable indicator."""
        if confidence >= 0.8:
            return "High Confidence"
        elif confidence >= 0.5:
            return "Medium Confidence"
        else:
            return "Low Confidence"
    
    @staticmethod
    def _format_citation(citation: Citation) -> str:
        """Format a single citation in a readable way."""
        parts = [
            f"[{citation.doc_title}]",
            f"(ID: {citation.doc_id})",
            f"Type: {citation.doc_type}"
        ]
        
        if citation.section:
            parts.append(f"Section: {citation.section}")
        
        if citation.version:
            parts.append(f"Version: {citation.version}")
            
        if citation.last_updated:
            try:
                date = datetime.fromisoformat(citation.last_updated)
                parts.append(f"Last Updated: {date.strftime('%Y-%m-%d')}")
            except ValueError:
                parts.append(f"Last Updated: {citation.last_updated}")
        
        confidence_indicator = AnswerFormatter._format_confidence_indicator(citation.confidence)
        parts.append(f"({confidence_indicator})")
        
        return " | ".join(parts)
    
    @staticmethod
    def format_answer(generated_answer: GeneratedAnswer) -> str:
        """Format the complete answer with citations and warnings."""
        parts = []
        
        # Add version warning if applicable
        if generated_answer.has_outdated_info:
            versions = ", ".join(generated_answer.outdated_versions)
            parts.append(f"⚠️ **Version Differences Detected**: This answer contains information from multiple versions: {versions}\n")
        
        # Add conflicting information warning if applicable
        if generated_answer.has_conflicting_info:
            parts.append("⚠️ **Note**: The available information contains some conflicts. Both perspectives are presented below.\n")
        
        # Add the main answer
        parts.append(generated_answer.answer_text)
        
        # Add citations section if there are any
        if generated_answer.citations:
            parts.append("\n\n**Sources:**")
            for citation in generated_answer.citations:
                parts.append(f"- {AnswerFormatter._format_citation(citation)}")
        
        # Add overall confidence score
        overall_confidence = AnswerFormatter._format_confidence_indicator(generated_answer.confidence_score)
        parts.append(f"\n**Overall Confidence**: {overall_confidence}")
        
        return "\n".join(parts) 
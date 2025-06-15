import re
from typing import List, Dict, Any

def extract_numbered_lists(text: str) -> List[Dict[str, Any]]:
    """
    Extract numbered lists and their context from text.
    Returns list of dicts with start/end positions and content.
    """
    # Match patterns like "1.", "1)", "(1)", etc.
    numbered_patterns = [
        r'\d+\.',  # 1.
        r'\d+\)',  # 1)
        r'\(\d+\)',  # (1)
    ]
    
    lists = []
    for pattern in numbered_patterns:
        # Find all sequences of numbered items
        matches = list(re.finditer(fr'(?m)^(?:\s*{pattern}\s+.+\n?)+', text, re.MULTILINE))
        for match in matches:
            lists.append({
                'start': match.start(),
                'end': match.end(),
                'content': match.group(),
                'type': 'numbered_list'
            })
    
    return sorted(lists, key=lambda x: x['start'])

def identify_section_boundaries(text: str) -> List[Dict[str, Any]]:
    """
    Identify logical section boundaries in text.
    Returns list of dicts with start/end positions and section type.
    """
    sections = []
    
    # Find headings (indicated by **Title** in markdown)
    heading_matches = list(re.finditer(r'\*\*(.*?)\*\*:', text))
    for i, match in enumerate(heading_matches):
        start = match.start()
        end = heading_matches[i + 1].start() if i < len(heading_matches) - 1 else len(text)
        sections.append({
            'start': start,
            'end': end,
            'content': text[start:end].strip(),
            'type': 'section'
        })
    
    return sections

def find_step_sequences(text: str) -> List[Dict[str, Any]]:
    """
    Identify step-by-step sequences in text.
    Returns list of dicts with start/end positions and steps content.
    """
    # Match common step indicators
    step_patterns = [
        r'Step \d+[:.)]',  # Step 1:, Step 2., Step 3)
        r'First,?|Second,?|Third,?|Finally,?',  # First, Second, etc.
        r'\d+\.\s+[A-Z]'  # 1. Start with...
    ]
    
    steps = []
    for pattern in step_patterns:
        matches = list(re.finditer(fr'(?m)^(?:\s*{pattern}\s+.+\n?)+', text, re.MULTILINE))
        for match in matches:
            steps.append({
                'start': match.start(),
                'end': match.end(),
                'content': match.group(),
                'type': 'steps'
            })
    
    return sorted(steps, key=lambda x: x['start'])

def merge_overlapping_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge overlapping text regions to prevent splitting related content.
    """
    if not regions:
        return []
    
    # Sort regions by start position
    sorted_regions = sorted(regions, key=lambda x: x['start'])
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        previous = merged[-1]
        
        # If current region overlaps with previous
        if current['start'] <= previous['end']:
            # Extend previous region if current ends later
            if current['end'] > previous['end']:
                previous['end'] = current['end']
                previous['content'] = previous['content'] + current['content']
        else:
            merged.append(current)
    
    return merged 
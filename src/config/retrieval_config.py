from datetime import timedelta

# Document type priorities (higher number = higher priority)
DOC_TYPE_PRIORITIES = {
    "product_doc": 3,
    "resolved_ticket": 2,
    "pending_ticket": 1
}

# Query classification patterns
QUERY_PATTERNS = {
    "authentication": [
        r"login",
        r"sign[- ]?in",
        r"auth(?:entication)?",
        r"credentials?"
    ],
    "synchronization": [
        r"sync(?:hronization)?(?:ing)?",
        r"data[- ]sync",
        r"file[- ]sync"
    ],
    "error": [
        r"error",
        r"issue",
        r"problem",
        r"fail(?:ure|ed|ing)?"
    ]
}

# Negation patterns
NEGATION_PATTERNS = [
    r"not",
    r"n't",
    r"cannot",
    r"can't",
    r"won't",
    r"isn't",
    r"doesn't",
    r"didn't",
    r"never",
    r"no"
]

# Recency scoring
RECENCY_SETTINGS = {
    "max_age": timedelta(days=365),  # Documents older than this get minimum recency score
    "recent_threshold": timedelta(days=30),  # Documents newer than this get maximum recency score
    "weight": 0.3  # Weight of recency in final score (0-1)
}

# Relevance weights
RELEVANCE_WEIGHTS = {
    "semantic_score": 0.4,
    "keyword_score": 0.3,
    "doc_priority": 0.2,
    "recency": 0.1
}

# Keyword matching settings
KEYWORD_SETTINGS = {
    "min_ngram": 1,
    "max_ngram": 3,
    "min_similarity": 0.8
}

# Re-ranking settings
RERANK_TOP_K = 20  # Number of initial results to re-rank
FINAL_RESULTS_K = 5  # Number of results to return after re-ranking 
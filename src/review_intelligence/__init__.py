"""Review Intelligence — NLP toolkit for analyzing customer reviews.

Built on Hugging Face Transformers. The public API grows chapter by chapter;
Chapter 1 exposes off-the-shelf ``pipeline()``-based analyzers.
"""

__version__ = "0.1.0"

from review_intelligence.baseline import (
    analyze_review,
    classify_topics,
    score_sentiment,
    summarize,
)

__all__ = [
    "analyze_review",
    "classify_topics",
    "score_sentiment",
    "summarize",
    "__version__",
]

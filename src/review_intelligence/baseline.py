"""Chapter 1 baseline: analyze customer reviews with off-the-shelf pipelines.

The Hugging Face ``pipeline()`` is a high-level wrapper that hides three steps:

    raw text  ->  [tokenizer]  ->  token ids  ->  [transformer model]
              ->  logits  ->  [post-processing]  ->  human-readable output

Each analyzer below is deliberately built on a different transformer
*architecture*, so this module doubles as a concrete tour of Chapter 1:

    * score_sentiment  -> ENCODER model (DistilBERT). Encoders read the whole
      sentence at once and are strong at *understanding* tasks like
      classification.
    * classify_topics  -> ENCODER model used for zero-shot classification
      (BART fine-tuned on NLI). We can assign our own topic labels with no
      training data at all.
    * summarize        -> ENCODER-DECODER model (DistilBART). The encoder reads
      the review, the decoder *generates* a shorter version.

Pipelines are created lazily and cached, because loading a model is slow
(weights are downloaded and read into memory once, then reused).
"""

from __future__ import annotations

from functools import lru_cache

from transformers import Pipeline, pipeline

# Pinning specific checkpoints (rather than letting the task pick a default)
# makes results reproducible and lets us reason about which architecture runs.
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

# Default topics for customer reviews. Zero-shot means these are *not* baked
# into the model — we can change them freely without retraining.
DEFAULT_TOPICS = [
    "shipping and delivery",
    "product quality",
    "price and value",
    "customer service",
    "ease of use",
]


@lru_cache(maxsize=None)
def _get_pipeline(task: str, model: str) -> Pipeline:
    """Build a pipeline once and reuse it (loading weights is expensive)."""
    return pipeline(task, model=model)


def score_sentiment(text: str) -> dict:
    """Classify a review as POSITIVE/NEGATIVE with a confidence score.

    Uses an *encoder* model (DistilBERT fine-tuned on SST-2). Returns e.g.
    ``{"label": "POSITIVE", "score": 0.9998}``.
    """
    clf = _get_pipeline("sentiment-analysis", SENTIMENT_MODEL)
    return clf(text)[0]


def classify_topics(text: str, topics: list[str] | None = None) -> dict:
    """Assign relevance scores over ``topics`` without any task-specific training.

    Zero-shot classification reframes "is this review about X?" as a natural
    language inference problem, which is why we can invent the labels at call
    time. Returns the topics sorted by score (most relevant first).
    """
    labels = topics if topics is not None else DEFAULT_TOPICS
    clf = _get_pipeline("zero-shot-classification", ZERO_SHOT_MODEL)
    result = clf(text, candidate_labels=labels)
    return {"labels": result["labels"], "scores": result["scores"]}


def summarize(text: str, max_length: int = 60, min_length: int = 10) -> str:
    """Condense a long review into a short summary via an encoder-decoder model."""
    summarizer = _get_pipeline("summarization", SUMMARIZATION_MODEL)
    out = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
    return out[0]["summary_text"]


def analyze_review(text: str, topics: list[str] | None = None) -> dict:
    """Run all three analyzers on a single review and return a combined report."""
    return {
        "sentiment": score_sentiment(text),
        "topics": classify_topics(text, topics),
        "summary": summarize(text),
    }

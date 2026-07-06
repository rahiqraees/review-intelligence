"""Integration smoke tests for the Chapter 1 pipeline analyzers.

These load real models, so they are slow and marked accordingly. Run them
locally with ``pytest`` (all tests) or skip them in CI with
``pytest -m "not slow"``. They verify shape and sanity (a clearly positive
review scores POSITIVE), not exact numbers.
"""

import pytest

from review_intelligence import classify_topics, score_sentiment, summarize


@pytest.mark.slow
def test_sentiment_detects_positive():
    result = score_sentiment("I love this product, it works perfectly!")
    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.9


@pytest.mark.slow
def test_sentiment_detects_negative():
    result = score_sentiment("Terrible quality, broke after one day. Waste of money.")
    assert result["label"] == "NEGATIVE"


@pytest.mark.slow
def test_classify_topics_returns_sorted_scores():
    result = classify_topics(
        "Delivery took forever and the package arrived damaged.",
        topics=["shipping and delivery", "price and value"],
    )
    assert result["labels"][0] == "shipping and delivery"
    assert result["scores"] == sorted(result["scores"], reverse=True)


@pytest.mark.slow
def test_summarize_shortens_text():
    long_review = (
        "This coffee maker has completely changed my mornings. The build quality "
        "is excellent, it brews quickly, and the app integration lets me schedule "
        "a fresh pot before I even get out of bed. Cleanup is simple too. "
    ) * 2
    summary = summarize(long_review)
    assert 0 < len(summary) < len(long_review)

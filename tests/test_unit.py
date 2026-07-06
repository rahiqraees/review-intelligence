"""Fast unit tests that never download a model.

These monkeypatch the pipeline factory with a stub, so they run in
milliseconds and are safe for CI. They cover our own logic — output shaping,
the default-topics fallback, and report composition — rather than model quality
(that is what the slow integration tests in test_baseline.py are for).
"""

from review_intelligence import baseline


def test_score_sentiment_unwraps_first_result(monkeypatch):
    # Pipelines return a list; score_sentiment should return the single dict.
    monkeypatch.setattr(
        baseline,
        "_get_pipeline",
        lambda task, model: (lambda text: [{"label": "POSITIVE", "score": 0.9}]),
    )
    assert baseline.score_sentiment("anything") == {"label": "POSITIVE", "score": 0.9}


def test_classify_topics_falls_back_to_default_topics(monkeypatch):
    captured = {}

    def fake_pipeline(task, model):
        def run(text, candidate_labels):
            captured["labels"] = candidate_labels
            n = len(candidate_labels)
            return {"labels": candidate_labels, "scores": [1.0 / n] * n}

        return run

    monkeypatch.setattr(baseline, "_get_pipeline", fake_pipeline)
    baseline.classify_topics("some review")
    assert captured["labels"] == baseline.DEFAULT_TOPICS


def test_classify_topics_uses_custom_topics(monkeypatch):
    captured = {}

    def fake_pipeline(task, model):
        def run(text, candidate_labels):
            captured["labels"] = candidate_labels
            return {"labels": candidate_labels, "scores": [0.6, 0.4]}

        return run

    monkeypatch.setattr(baseline, "_get_pipeline", fake_pipeline)
    baseline.classify_topics("some review", topics=["a", "b"])
    assert captured["labels"] == ["a", "b"]


def test_analyze_review_composes_all_three(monkeypatch):
    monkeypatch.setattr(baseline, "score_sentiment", lambda t: {"label": "NEGATIVE", "score": 0.5})
    monkeypatch.setattr(baseline, "classify_topics", lambda t, topics=None: {"labels": ["x"], "scores": [1.0]})
    monkeypatch.setattr(baseline, "summarize", lambda t: "a summary")

    report = baseline.analyze_review("some review")
    assert set(report) == {"sentiment", "topics", "summary"}
    assert report["summary"] == "a summary"

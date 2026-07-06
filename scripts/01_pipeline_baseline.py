"""Chapter 1 demo: run the pipeline-based analyzers on sample customer reviews.

Run from the repo root (with the conda env active):

    python scripts/01_pipeline_baseline.py

The first run downloads model weights from the Hugging Face Hub (a few hundred
MB) and caches them locally; later runs are fast.
"""

from review_intelligence import analyze_review

SAMPLE_REVIEWS = [
    # Clearly positive, mentions quality + service.
    "Absolutely love this blender! It crushes ice in seconds and the customer "
    "support team replaced a faulty lid within two days. Worth every penny.",
    # Negative, mostly about shipping.
    "The product itself is fine, but it took three weeks to arrive and the box "
    "was crushed. I won't order from here again if delivery is this slow.",
    # Mixed — good product, bad price.
    "Great sound quality and the battery lasts all day. That said, $250 feels "
    "steep for what you get, and setup was more confusing than it should be.",
]


def main() -> None:
    for i, review in enumerate(SAMPLE_REVIEWS, start=1):
        report = analyze_review(review)
        sentiment = report["sentiment"]
        top_topic = report["topics"]["labels"][0]
        top_score = report["topics"]["scores"][0]

        print(f"\n=== Review {i} ===")
        print(review)
        print(f"  Sentiment : {sentiment['label']} ({sentiment['score']:.1%})")
        print(f"  Top topic : {top_topic} ({top_score:.1%})")
        print(f"  Summary   : {report['summary']}")


if __name__ == "__main__":
    main()

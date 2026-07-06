"""Chapter 3 demo: fine-tune a product-review sentiment classifier.

Smoke test (fast, tiny — just checks the pipeline runs):
    python scripts/03_finetune_sentiment.py --n-train 100 --n-eval 50 --epochs 1

Full run:
    python scripts/03_finetune_sentiment.py --n-train 4000 --n-eval 1000 --epochs 2

Writes the fine-tuned model to models/ (gitignored) and the metrics to
results/sentiment_finetune.json (committed as evidence of the result).
"""

import argparse
import json
from pathlib import Path

from review_intelligence.train import finetune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output-dir", default="models/distilbert-amazon-sentiment")
    parser.add_argument("--metrics-path", default="results/sentiment_finetune.json")
    args = parser.parse_args()

    metrics = finetune(
        n_train=args.n_train,
        n_eval=args.n_eval,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )

    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))

    ba, fa = metrics["baseline_accuracy"], metrics["finetuned_accuracy"]
    print("\n" + "=" * 48)
    print("RESULTS on product-review test set")
    print(f"  baseline (movie-trained) accuracy : {ba:.1%}")
    print(f"  fine-tuned (ours)        accuracy : {fa:.1%}")
    print(f"  improvement                       : {fa - ba:+.1%}")
    print("=" * 48)


if __name__ == "__main__":
    main()

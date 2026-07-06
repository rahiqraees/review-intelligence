# Review Intelligence

[![CI](https://github.com/rahiqraees/review-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rahiqraees/review-intelligence/actions/workflows/ci.yml)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/rahiqr/review-intelligence)

An NLP toolkit for turning unstructured customer reviews into structured,
actionable signal — sentiment, topics, and summaries — built on
[Hugging Face Transformers](https://huggingface.co/docs/transformers).

**▶ Try the live demo:** https://huggingface.co/spaces/rahiqr/review-intelligence
(first load is slow while models warm up, then fast).

The project is developed as one evolving system rather than a set of disconnected
notebooks: it starts with off-the-shelf transformer pipelines and grows toward a
fine-tuned, deployed model, engineered like production software (packaging,
tests, CI, and reproducible environments).

## Why this project

Customer feedback arrives as free text at a scale no team can read manually.
Review Intelligence classifies each review's **sentiment**, identifies which
**topics** it touches (shipping, quality, price, support, ...), and produces a
short **summary** — the building blocks of a feedback dashboard that tells a
business *what* customers are unhappy about and *how much*.

## What it does today

| Capability | Task | Transformer architecture |
|---|---|---|
| Sentiment scoring | text classification | Encoder (DistilBERT / SST-2) |
| Topic detection | zero-shot classification | Encoder (BART / MNLI) |
| Review summarization | summarization | Encoder–decoder (DistilBART) |

```python
from review_intelligence import analyze_review

analyze_review("Took three weeks to arrive and the box was crushed.")
# {'sentiment': {'label': 'NEGATIVE', 'score': 0.99...},
#  'topics':    {'labels': ['shipping and delivery', ...], 'scores': [...]},
#  'summary':   '...'}
```

## Quickstart

```bash
# 1. Create the environment (installs PyTorch + Transformers)
conda env create -f environment.yml
conda activate review-intelligence

# 2. Install the package
pip install -e .

# 3. Run the Chapter 1 demo on sample reviews
python scripts/01_pipeline_baseline.py

# 4. Run the tests
pytest
```

The first run downloads model weights from the Hugging Face Hub and caches them
locally; subsequent runs are fast.

## Fine-tuning result

The Chapter 1 sentiment model was trained on movie reviews (SST-2). To fit our
actual domain, a DistilBERT classifier was fine-tuned on Amazon **product**
reviews and benchmarked against the movie-trained model on the same 1,000
held-out product reviews:

| Model | Accuracy | F1 |
|---|---|---|
| Baseline (movie-trained, SST-2) | 88.6% | 0.888 |
| **Fine-tuned (product reviews)** | **89.7%** | **0.899** |

The modest gain is itself the finding: binary sentiment transfers well across
domains, so a strong off-the-shelf baseline is hard to beat by a wide margin.
Full numbers in [`results/sentiment_finetune.json`](results/sentiment_finetune.json).

The fine-tuned model is published at
[`rahiqr/distilbert-amazon-sentiment`](https://huggingface.co/rahiqr/distilbert-amazon-sentiment)
and powers the sentiment scoring in the [live demo](https://huggingface.co/spaces/rahiqr/review-intelligence).

```bash
# Reproduce (fine-tunes on a streamed subset; ~10 min on Apple Silicon)
python scripts/03_finetune_sentiment.py --n-train 6000 --n-eval 1000 --epochs 2
```

## Project structure

```
src/review_intelligence/   # installable package (the reusable API)
scripts/                   # runnable demos, one per milestone
tests/                     # pytest suite
notebooks/                 # exploration and write-ups
data/                      # datasets (gitignored; regenerated from code)
```

## Roadmap

- [x] **Baseline** — analyze reviews with pretrained `pipeline()` models
- [x] **Behind the pipeline** — tokenization and model internals
- [x] **Fine-tuning** — train a custom review classifier and evaluate it
- [ ] **Model sharing** — publish the fine-tuned model to the Hugging Face Hub
- [ ] **Data & tokenizers** — scalable dataset processing
- [x] **Deployment** — interactive Gradio demo on Hugging Face Spaces

## Tech stack

Python · PyTorch · Hugging Face Transformers & Datasets · pytest

## License

MIT

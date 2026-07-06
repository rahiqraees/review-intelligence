"""Chapter 3: fine-tune a sentiment classifier on real product reviews.

Our Chapter 1 model was trained on SST-2 (movie reviews). Here we fine-tune
DistilBERT on Amazon *product* reviews so it fits our actual domain, then
measure the improvement against the movie-trained model on the same test set.

The pieces mirror the HF course Chapter 3:
    load data (datasets) -> tokenize (.map) -> Trainer (TrainingArguments) -> evaluate
"""

from __future__ import annotations

import itertools

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Base model we fine-tune (no task head yet — we add a fresh one).
BASE_MODEL = "distilbert-base-uncased"
# The Chapter 1 movie-trained model we use as the baseline to beat.
BASELINE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# amazon_polarity labels: 0 = negative, 1 = positive (matches the SST-2 model).
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}


def load_review_subset(n_train: int, n_eval: int, seed: int = 42):
    """Stream a balanced subset of Amazon product reviews (no full download).

    Streaming lets us grab a few thousand rows without pulling the whole 1.6 GB
    dataset. We shuffle the stream so classes are mixed, and draw the eval set
    from the *test* split so it never overlaps with training data.
    """
    def take(split: str, n: int) -> Dataset:
        stream = load_stream(split).shuffle(seed=seed, buffer_size=10_000)
        rows = list(itertools.islice(stream, n))
        return Dataset.from_list(
            [{"text": r["content"], "label": r["label"]} for r in rows]
        )

    return take("train", n_train), take("test", n_eval)


def load_stream(split: str):
    from datasets import load_dataset

    return load_dataset("amazon_polarity", split=split, streaming=True)


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int = 256) -> Dataset:
    """Turn raw text into model inputs for every row at once.

    ``.map(batched=True)`` runs the tokenizer over chunks of rows efficiently.
    We keep sequences variable-length here and let the data collator pad each
    batch to its own longest row at training time (dynamic padding).
    """
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = ds.map(_tok, batched=True, remove_columns=["text"])
    return tokenized.rename_column("label", "labels")


def compute_metrics(eval_pred):
    """Accuracy + F1 from the Trainer's (logits, labels) tuple."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def build_trainer(model, train_ds, eval_ds, tokenizer, output_dir: str, epochs: int):
    """Wire up TrainingArguments + Trainer with dynamic padding and metrics."""
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,          # small LR: we're nudging pretrained weights
        eval_strategy="epoch",       # evaluate on the eval set after each epoch
        save_strategy="no",          # skip intermediate checkpoints (save at end)
        logging_steps=25,
        report_to="none",            # no external experiment tracker
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )


def evaluate_baseline(eval_ds, tokenizer) -> dict:
    """Score the movie-trained (SST-2) model on our product-review eval set."""
    model = AutoModelForSequenceClassification.from_pretrained(BASELINE_MODEL)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="models/_baseline_tmp", report_to="none"),
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate()


def finetune(n_train: int, n_eval: int, epochs: int, output_dir: str) -> dict:
    """End-to-end: load data, baseline, fine-tune, evaluate, save. Returns metrics."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    raw_train, raw_eval = load_review_subset(n_train, n_eval)
    train_ds = tokenize_dataset(raw_train, tokenizer)
    eval_ds = tokenize_dataset(raw_eval, tokenizer)

    baseline = evaluate_baseline(eval_ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    trainer = build_trainer(model, train_ds, eval_ds, tokenizer, output_dir, epochs)
    trainer.train()
    finetuned = trainer.evaluate()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "n_train": n_train,
        "n_eval": n_eval,
        "epochs": epochs,
        "baseline_accuracy": baseline["eval_accuracy"],
        "baseline_f1": baseline["eval_f1"],
        "finetuned_accuracy": finetuned["eval_accuracy"],
        "finetuned_f1": finetuned["eval_f1"],
    }

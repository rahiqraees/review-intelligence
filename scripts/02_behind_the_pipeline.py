"""Chapter 2 walkthrough: reproduce score_sentiment() by hand.

In Chapter 1 we called ``pipeline("sentiment-analysis")`` and got a label back.
That one call actually runs three stages:

    text -> [tokenizer] -> input_ids + attention_mask
         -> [model]     -> logits
         -> [softmax]   -> probabilities -> label

Here we run each stage ourselves so nothing is a black box. At the end we
compare our manual result to the Chapter 1 pipeline and confirm they match.

Run:  python scripts/02_behind_the_pipeline.py
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from review_intelligence.baseline import SENTIMENT_MODEL, score_sentiment

REVIEW = "The build quality is fantastic and it arrived early."


def main() -> None:
    # ------------------------------------------------------------------
    # Load the two halves the pipeline bundled together: a tokenizer and a
    # model. `Auto*` classes read the checkpoint's config and pick the right
    # concrete class for us (here: a DistilBERT tokenizer + classifier).
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    print(f"Review: {REVIEW}\n")

    # ---- STAGE 1: tokenize -------------------------------------------------
    # return_tensors="pt" gives us PyTorch tensors ready for the model.
    inputs = tokenizer(REVIEW, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print("STAGE 1 — TOKENIZER")
    print(f"  tokens        : {tokens}")
    print(f"  input_ids     : {input_ids[0].tolist()}")
    print(f"  attention_mask: {inputs['attention_mask'][0].tolist()}")
    # Round-trip back to text to see what the ids mean:
    print(f"  decoded       : {tokenizer.decode(input_ids[0])}\n")

    # ---- STAGE 2: model forward pass --------------------------------------
    # torch.no_grad() disables gradient tracking — we're only doing inference,
    # so we don't need the machinery used for training (faster, less memory).
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # raw, unnormalized scores (one per class)

    print("STAGE 2 — MODEL")
    print(f"  logits shape  : {tuple(logits.shape)}  (batch_size, num_labels)")
    print(f"  logits        : {logits[0].tolist()}\n")

    # ---- STAGE 3: post-process --------------------------------------------
    # Softmax turns the two logits into probabilities that sum to 1.
    probs = torch.softmax(logits, dim=-1)[0]
    # The model config stores which index means which label.
    id2label = model.config.id2label
    predicted_id = int(torch.argmax(probs))

    print("STAGE 3 — POST-PROCESS (softmax + argmax)")
    for i, p in enumerate(probs.tolist()):
        print(f"  {id2label[i]:>8}: {p:.4f}")
    print(f"  => prediction : {id2label[predicted_id]} ({probs[predicted_id]:.4f})\n")

    # ---- Sanity check: does this match the Chapter 1 pipeline? -------------
    print("CHECK — Chapter 1 pipeline on the same text")
    print(f"  {score_sentiment(REVIEW)}")


if __name__ == "__main__":
    main()

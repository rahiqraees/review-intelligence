"""Gradio demo for Review Intelligence.

Paste a customer review and get sentiment, topic relevance, and a summary —
the three pipeline analyzers from the package, behind a web UI.

Run locally:   python app.py
On Hugging Face Spaces this file is the entry point (SDK: gradio).
"""

import gradio as gr

from review_intelligence import analyze_review
from review_intelligence.baseline import DEFAULT_TOPICS

EXAMPLES = [
    "Absolutely love this blender! It crushes ice in seconds and the customer "
    "support team replaced a faulty lid within two days. Worth every penny.",
    "The product itself is fine, but it took three weeks to arrive and the box "
    "was crushed. I won't order from here again if delivery is this slow.",
    "Great sound quality and the battery lasts all day. That said, $250 feels "
    "steep for what you get, and setup was more confusing than it should be.",
]


def analyze(review: str, topics_text: str):
    """Run the analyzers and shape the output for the Gradio components."""
    if not review or not review.strip():
        return {}, {}, ""

    # Comma-separated custom topics override the defaults; blank falls back.
    topics = [t.strip() for t in topics_text.split(",") if t.strip()] or None
    report = analyze_review(review, topics=topics)

    sentiment = {report["sentiment"]["label"]: report["sentiment"]["score"]}
    topics_out = dict(zip(report["topics"]["labels"], report["topics"]["scores"]))
    return sentiment, topics_out, report["summary"]


with gr.Blocks(title="Review Intelligence") as demo:
    gr.Markdown(
        "# Review Intelligence\n"
        "Turn a customer review into structured signal — **sentiment**, "
        "**topics**, and a **summary** — using Hugging Face Transformers."
    )
    with gr.Row():
        with gr.Column():
            review_in = gr.Textbox(
                label="Customer review", lines=6, placeholder="Paste a review..."
            )
            topics_in = gr.Textbox(
                label="Topics (comma-separated, optional)",
                placeholder=", ".join(DEFAULT_TOPICS),
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            sentiment_out = gr.Label(label="Sentiment")
            topics_out = gr.Label(label="Topic relevance")
            summary_out = gr.Textbox(label="Summary", lines=3)

    analyze_btn.click(
        analyze,
        inputs=[review_in, topics_in],
        outputs=[sentiment_out, topics_out, summary_out],
    )
    gr.Examples(examples=[[e, ""] for e in EXAMPLES], inputs=[review_in, topics_in])


if __name__ == "__main__":
    demo.launch()

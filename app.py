"""
Milestone 5 — Gradio query interface for the ASU Parking & Transit Unofficial Guide.

Run:
    python app.py
Then open http://localhost:7860

Shows, for each question:
  - the grounded answer (from retrieved context only)
  - the source document(s) it drew from (programmatic attribution)
  - the actual retrieved chunks + distance scores, so a viewer can SEE that the
    answer is grounded in real retrieved text, not the model's training data.
"""

from __future__ import annotations

import gradio as gr

from query import ask

EXAMPLES = [
    "Which parking structure fills up the earliest, and where can I park late morning?",
    "What are free or cheap ways to park near campus without buying a permit?",
    "Can I appeal a parking ticket if the plate reader misread my plate?",
    "Is the U-Pass worth it for commuting from Mesa?",
    "Should I rely on the light rail instead of driving to ASU?",
    "What time does the dining hall close?",  # out-of-domain → should decline
]


def handle_query(question: str):
    question = (question or "").strip()
    if not question:
        return "Type a question above and press Ask.", "", ""

    result = ask(question)

    sources = "\n".join(f"• {s}" for s in result["sources"]) or \
        "(none — the documents don't cover this)"

    # Show the retrieved evidence so grounding is visible in the demo.
    retrieved_md = []
    for i, h in enumerate(result["hits"], start=1):
        body = h["text"].split("\n", 1)[-1].strip()
        retrieved_md.append(
            f"**[{i}]** `{h['source']}` — distance **{h['distance']:.3f}**\n\n> "
            + body.replace("\n", "\n> ")
        )
    retrieved = "\n\n---\n\n".join(retrieved_md)

    return result["answer"], sources, retrieved


with gr.Blocks(title="ASU Parking & Transit — Unofficial Guide") as demo:
    gr.Markdown(
        "# 🅿️ ASU Parking & Transit — The Unofficial Guide\n"
        "Ask a plain-language question. Answers are grounded **only** in real "
        "student posts and reviews that were retrieved — with sources cited. "
        "If the documents don't cover your question, the system says so instead "
        "of guessing."
    )
    with gr.Row():
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. Which structure fills up earliest?",
            scale=4,
            autofocus=True,
        )
        btn = gr.Button("Ask", variant="primary", scale=1)

    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Sources (documents cited)", lines=3)
    with gr.Accordion("Retrieved chunks (the evidence behind the answer)", open=False):
        retrieved = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources, retrieved])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources, retrieved])


if __name__ == "__main__":
    demo.launch()

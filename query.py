"""
Milestone 5 — Grounded answer generation.

    user query -> retrieve(top-k chunks) -> Groq llama-3.3-70b-versatile -> grounded answer
                                                                          -> sources (programmatic)

Grounding design (the core engineering challenge per the spec):
  - The system prompt ENFORCES answering only from the supplied context and to
    reply with a fixed refusal string when the context is insufficient.
  - Each context block is labelled [1], [2], ... with its source filename, and
    the model is told to cite those bracket numbers inline.
  - Source attribution is NOT left to the model: `ask()` returns the list of
    source files for the retrieved chunks programmatically, so a citation is
    guaranteed even if the model forgets to add one.

Usage:
    python query.py "your question here"      # one-off CLI
    python query.py                            # runs the M5 grounding test suite
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from embed import retrieve

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
TOP_K = 4
NO_ANSWER = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are an assistant that answers questions about ASU Tempe \
parking and transit using ONLY the student-written context provided by the user.

Rules — follow exactly:
1. Use ONLY facts contained in the numbered context passages. Do NOT use any \
outside or general knowledge, even if you are confident it is correct.
2. If the context does not contain enough information to answer the question, \
reply with exactly this sentence and nothing else: "{NO_ANSWER}"
3. Cite the passages you used with their bracket numbers inline, e.g. [1], [2].
4. These are informal student opinions, not official policy. When sources \
disagree, say so and present both sides rather than picking one.
5. Be concise and specific. Quote concrete details (prices, times, locations) \
that appear in the context."""

USER_TEMPLATE = """Question: {question}

Context passages:
{context}

Answer the question using only the context above. Cite passages by their number."""


def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_key_here":
        raise SystemExit("GROQ_API_KEY is not set in .env")
    return Groq(api_key=key)


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as numbered, source-labelled passages."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        body = h["text"].split("\n", 1)[-1].strip()  # drop the [source] prefix line
        blocks.append(f"[{i}] (source: {h['source']})\n{body}")
    return "\n\n".join(blocks)


def ask(question: str, k: int = TOP_K) -> dict:
    """End-to-end: retrieve -> generate grounded answer -> attach sources.

    Returns {answer, sources, hits} where `sources` is the de-duplicated list of
    source files actually retrieved (programmatic attribution).
    """
    hits = retrieve(question, k=k)

    # De-duplicated source list, preserving retrieval order — guaranteed citation.
    sources: list[str] = []
    for h in hits:
        if h["source"] not in sources:
            sources.append(h["source"])

    if not hits:
        return {"answer": NO_ANSWER, "sources": [], "hits": []}

    context = format_context(hits)
    resp = _client().chat.completions.create(
        model=MODEL,
        temperature=0,            # deterministic, reduces drift from context
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, context=context)},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    # If the model declined, don't imply the sources supported an answer.
    cited_sources = [] if answer.strip().startswith(NO_ANSWER) else sources
    return {"answer": answer, "sources": cited_sources, "hits": hits}


# ---------------------------------------------------------------------------
# M5 grounding test suite
# ---------------------------------------------------------------------------
TESTS = [
    "Which parking structure fills up the earliest, and where can I still find a spot late morning?",
    "Can I appeal a parking ticket if the plate reader misread my license plate?",
    "What are free or cheap ways to park near campus without a permit?",
    # Out-of-domain — must trigger the refusal, not a hallucinated answer.
    "What time does the ASU dining hall close on weekends?",
]


def _run_tests() -> None:
    for q in TESTS:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        result = ask(q)
        print(result["answer"])
        print("\nSources:", ", ".join(result["sources"]) or "(none — declined)")
        print()


def main() -> None:
    if len(sys.argv) > 1:
        result = ask(" ".join(sys.argv[1:]))
        print(result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f"  • {s}")
        if not result["sources"]:
            print("  (none — not enough information in the documents)")
    else:
        _run_tests()


if __name__ == "__main__":
    main()

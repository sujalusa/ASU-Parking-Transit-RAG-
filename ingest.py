"""
Milestone 3 — Document ingestion + chunking pipeline.

Two jobs (per the project spec):
  1. Load every document in documents/ into memory and clean it.
  2. Split each cleaned document into chunks the embedding model can work with.

Design follows planning.md:
  - Chunk size ~500 chars, overlap ~80 (~16%), boundary-aware
    (RecursiveCharacterTextSplitter, separators preferring blank lines / comment
    boundaries) so a chunk ≈ one self-contained opinion.
  - Each document begins with a small header (Source:/Title:/Type:/Topic:). That
    header is parsed into METADATA and stripped from the body, then a short context
    line is prepended to each chunk so a review like "fills up fast" still records
    *which* garage it's about (the standalone-meaning test).

Run:  python ingest.py
Output artifacts (consistent format for Milestone 4):
  - data/raw_documents.json   (cleaned full docs + metadata)
  - data/chunks.json          (chunks + metadata, ready to embed)
"""

from __future__ import annotations

import html
import json
import random
import re
import statistics
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Config — mirrors planning.md "Chunking Strategy"
# ----------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "documents"
DATA_DIR = Path(__file__).parent / "data"
CHUNK_SIZE = 500          # characters (~100-125 tokens, under MiniLM's 256-token cap)
CHUNK_OVERLAP = 80        # ~16%
MIN_CHUNK_CHARS = 80      # merge anything shorter (e.g. a stranded username line)
RANDOM_SEED = 13          # deterministic "random" sample for inspection

# Header lines we treat as metadata, not chunk content.
HEADER_KEYS = ("Source:", "Title:", "Type:", "Topic:")

# Reddit / web boilerplate to strip during cleaning. Even though most of the
# corpus is already clean, this makes the pipeline robust to raw pasted threads.
BOILERPLATE_LINES = {
    "upvote", "downvote", "reply", "award", "share", "report", "save",
    "read more", "see more", "show more replies", "continue this thread",
    "more replies", "load more comments",
}
# A standalone line that is just a vote count / "• 1y ago" timestamp, etc.
VOTE_COUNT_RE = re.compile(r"^\d+(\.\d+)?[kK]?$")
TIMESTAMP_RE = re.compile(r"^[•·]\s*\d+\s*(mo|y|d|h|m)\s*ago", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------------
# Stage 1 — load + parse header
# ----------------------------------------------------------------------------
def parse_document(path: Path) -> dict:
    """Read one .txt file, split its header lines from the body."""
    raw = path.read_text(encoding="utf-8")
    meta = {
        "source": path.name,          # filename — used for citations later
        "source_type": "",            # value of "Source:" (e.g. r/ASU, Google Reviews)
        "title": "",
        "topic": "",
    }
    body_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source:"):
            meta["source_type"] = stripped[len("Source:"):].strip()
        elif stripped.startswith("Title:"):
            meta["title"] = stripped[len("Title:"):].strip()
        elif stripped.startswith("Topic:"):
            meta["topic"] = stripped[len("Topic:"):].strip()
        elif stripped.startswith("Type:"):
            continue  # captured implicitly; not needed downstream
        else:
            body_lines.append(line)
    meta["body"] = "\n".join(body_lines).strip()
    return meta


def load_documents() -> list[dict]:
    docs = [parse_document(p) for p in sorted(DOCS_DIR.glob("*.txt"))]
    if not docs:
        raise SystemExit(f"No .txt files found in {DOCS_DIR}")
    return docs


# ----------------------------------------------------------------------------
# Stage 2 — clean
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove HTML, entities, and Reddit/web boilerplate; normalize whitespace."""
    text = html.unescape(text)            # &amp; -> &, &#39; -> '
    text = HTML_TAG_RE.sub("", text)      # strip any <tag>

    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if not stripped:
            kept.append("")              # preserve blank lines (paragraph boundaries)
            continue
        if low in BOILERPLATE_LINES:
            continue
        if VOTE_COUNT_RE.match(stripped):
            continue
        if TIMESTAMP_RE.match(stripped):
            continue
        kept.append(stripped)

    # Collapse 3+ blank lines down to a single paragraph break.
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ----------------------------------------------------------------------------
# Stage 3 — chunk
# ----------------------------------------------------------------------------
def make_splitter() -> RecursiveCharacterTextSplitter:
    # Prefer blank-line (comment/review) boundaries, then newline, then sentence,
    # then word — only cut mid-word as a last resort.
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )


def context_prefix(meta: dict) -> str:
    """Short standalone-meaning header prepended to each chunk.

    A Google review chunk that just says 'fills up fast' loses meaning without
    knowing which garage. The prefix records the source so the chunk is
    answerable on its own and the LLM can attribute it.
    """
    title = meta["title"] or meta["source"]
    src = meta["source_type"] or "unknown source"
    return f"[{src} — {title}]\n"


def merge_small_pieces(pieces: list[str]) -> list[str]:
    """Fold fragments shorter than MIN_CHUNK_CHARS into a neighbor.

    The recursive splitter can strand a short line (e.g. a Reddit username
    'Longjumping-Pass2825:') as its own piece. Such a chunk has no standalone
    meaning, so we merge it forward into the next piece (or backward if last).
    """
    merged: list[str] = []
    carry = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        combined = f"{carry}\n{piece}".strip() if carry else piece
        if len(combined) < MIN_CHUNK_CHARS:
            carry = combined          # still too short — keep accumulating
        else:
            merged.append(combined)
            carry = ""
    if carry:                          # trailing fragment — attach to last chunk
        if merged:
            merged[-1] = f"{merged[-1]}\n{carry}".strip()
        else:
            merged.append(carry)
    return merged


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = make_splitter()
    chunks: list[dict] = []
    for meta in docs:
        cleaned = clean_text(meta["body"])
        prefix = context_prefix(meta)
        pieces = merge_small_pieces(splitter.split_text(cleaned))
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:                # drop empty chunks (spec checkpoint guard)
                continue
            chunks.append({
                "id": f"{meta['source']}::chunk_{i}",
                "text": prefix + piece,  # embedding-ready text (self-contained)
                "body_only": piece,      # raw chunk without the context prefix
                "source": meta["source"],
                "source_type": meta["source_type"],
                "title": meta["title"],
                "topic": meta["topic"],
                "chunk_index": i,
            })
    return chunks


# ----------------------------------------------------------------------------
# Inspection / verification (do NOT skip — spec checkpoint)
# ----------------------------------------------------------------------------
def inspect(docs: list[dict], chunks: list[dict]) -> None:
    print("=" * 78)
    print("STAGE 2 CHECK — one cleaned document (read it for leftover junk)")
    print("=" * 78)
    sample_doc = docs[0]
    print(f"source: {sample_doc['source']}  |  type: {sample_doc['source_type']}")
    print("-" * 78)
    print(clean_text(sample_doc["body"])[:700])
    print("...\n")

    print("=" * 78)
    print(f"STAGE 3 CHECK — {len(chunks)} total chunks from {len(docs)} documents")
    print("=" * 78)

    lengths = [len(c["body_only"]) for c in chunks]
    print(f"chunk length (chars): min={min(lengths)}  "
          f"median={int(statistics.median(lengths))}  "
          f"mean={int(statistics.mean(lengths))}  max={max(lengths)}")
    empties = sum(1 for c in chunks if not c["body_only"].strip())
    print(f"empty chunks: {empties}  (must be 0)")

    floor_ok = "OK" if len(chunks) >= 50 else "TOO FEW (<50 → chunks too large?)"
    ceil_ok = "OK" if len(chunks) <= 2000 else "TOO MANY (>2000 → chunks too small?)"
    print(f"count vs rubric floor (>=50): {floor_ok}")
    print(f"count vs rubric ceiling (<=2000): {ceil_ok}\n")

    print("-" * 78)
    print("5 RANDOM CHUNKS — each should be readable, substantive, self-contained")
    print("-" * 78)
    rng = random.Random(RANDOM_SEED)
    for n, c in enumerate(rng.sample(chunks, k=min(5, len(chunks))), start=1):
        print(f"\n[{n}] id={c['id']}  ({len(c['body_only'])} chars)  source={c['source']}")
        print(c["text"])


# ----------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    docs = load_documents()
    chunks = chunk_documents(docs)

    # Save consistent-format artifacts for Milestone 4.
    raw_out = [{k: v for k, v in d.items()} for d in docs]
    (DATA_DIR / "raw_documents.json").write_text(
        json.dumps(raw_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    inspect(docs, chunks)
    print(f"\nWrote {len(docs)} docs -> data/raw_documents.json")
    print(f"Wrote {len(chunks)} chunks -> data/chunks.json")


if __name__ == "__main__":
    main()

"""
Milestone 4 — Embed chunks into ChromaDB + retrieval.

Pipeline position (see planning.md Architecture):
    chunks.json  ->  all-MiniLM-L6-v2 embeddings  ->  ChromaDB (persistent)  ->  retrieve(query, k)

- Embedding model: all-MiniLM-L6-v2 (sentence-transformers), local, 384-dim.
- Vector store: ChromaDB persistent client on disk (data/chroma/).
- Each chunk is stored with metadata for attribution: source filename,
  source_type, title, topic, chunk_index.

Usage:
    python embed.py            # (re)build the collection from data/chunks.json
    python embed.py --test     # build, then run sample eval queries and print results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
CHUNKS_PATH = BASE / "data" / "chunks.json"
CHROMA_DIR = BASE / "data" / "chroma"
COLLECTION_NAME = "asu_parking_transit"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine distance: 0 = identical meaning, ~1 = unrelated. The MiniLM default
# space in Chroma is L2; we set cosine explicitly so distances are comparable
# to the < 0.5 target in the spec.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def _client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_collection() -> int:
    """Embed every chunk and (re)load it into a fresh ChromaDB collection."""
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit(f"No chunks in {CHUNKS_PATH}; run ingest.py first.")

    client = _client()
    # Start clean so re-runs don't duplicate vectors.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # use cosine distance
    )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]          # text incl. context prefix
    metadatas = [
        {
            "source": c["source"],                   # filename — for citation
            "source_type": c["source_type"],
            "title": c["title"],
            "topic": c["topic"],
            "chunk_index": c["chunk_index"],          # position in source doc
        }
        for c in chunks
    ]
    embeddings = embed_texts(documents)

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return collection.count()


def get_collection():
    """Open the existing persistent collection (read path used by retrieval)."""
    return _client().get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return the top-k most similar chunks with source info + distance."""
    collection = get_collection()
    q_emb = embed_texts([query])
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "source_type": meta["source_type"],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return hits


# ---------------------------------------------------------------------------
# Retrieval smoke test — 3+ of the planning.md eval questions
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "Which parking structure fills up the earliest and how do I find a spot late morning?",
    "What are free or cheap ways to park near campus without buying a permit?",
    "Can I appeal a parking ticket if the plate reader misread my license plate?",
    "Is the U-Pass worth it for commuting from Mesa and how reliable is the light rail?",
]


def run_tests(k: int = 4) -> None:
    for q in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: {q}")
        print("=" * 80)
        for rank, hit in enumerate(retrieve(q, k=k), start=1):
            flag = "  <-- weak (>0.5)" if hit["distance"] > 0.5 else ""
            print(f"\n[{rank}] distance={hit['distance']:.3f}{flag}  "
                  f"source={hit['source']}  (chunk {hit['chunk_index']})")
            body = hit["text"].split("\n", 1)[-1]  # drop the [source] prefix line
            print("    " + body[:300].replace("\n", "\n    "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run sample queries after building")
    parser.add_argument("-k", type=int, default=4, help="top-k for the test queries")
    args = parser.parse_args()

    count = build_collection()
    print(f"Embedded and stored {count} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' (cosine, model={EMBED_MODEL_NAME}).\n")

    if args.test:
        run_tests(k=args.k)


if __name__ == "__main__":
    main()

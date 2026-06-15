# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language questions
about **ASU Tempe parking & transit** using real student-written knowledge —
Reddit threads and Google reviews — instead of the official parking website.

---

## Domain

**ASU Tempe parking & transit.** This system makes searchable the parking and
commuting knowledge ASU students actually share with each other: which structures
fill by 8am, whether the ~$720 gold permit is worth it, free park-and-ride and
visitor-pass workarounds, light rail reliability and safety, and how to win a
citation appeal.

This knowledge is valuable because it is **experiential and constantly changing**,
and it is **hard to find through official channels**. The official ASU parking site
(cfo.asu.edu) lists permit rates and rules, but it will never tell you that the roof
level of the Rural Rd garage is usually open even at 10am, that the LDS institute
hands out $5 parking passes, that you can park free at the Tempe Public Library and
ride the Orbit bus in, or that a plate-reader misread is a winnable appeal. That
information lives scattered across r/ASU threads, Google reviews of individual
garages, and word of mouth — exactly the "unofficial guide" this project assembles.

---

## Running the System

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. API key — copy the example and paste your free Groq key
cp .env.example .env          # then set GROQ_API_KEY=gsk_...  (https://console.groq.com)

# 3. Build the pipeline
python ingest.py              # load + clean + chunk documents/  -> data/chunks.json
python embed.py               # embed chunks -> ChromaDB (data/chroma/)

# 4a. Ask from the command line
python query.py "Which structure fills up earliest?"

# 4b. Or launch the web UI
python app.py                 # open http://localhost:7860
```

| Stage | File | What it does |
|-------|------|--------------|
| Ingestion + chunking | `ingest.py` | loads `documents/*.txt`, cleans, chunks → `data/chunks.json` |
| Embedding + vector store | `embed.py` | embeds chunks with MiniLM, stores in ChromaDB, `retrieve(query, k)` |
| Grounded generation | `query.py` | `ask(question)` → retrieve → Groq Llama-3.3 → grounded, cited answer |
| Interface | `app.py` | Gradio web UI showing answer, sources, and the retrieved evidence |

---

## Document Sources

13 documents across **two source types** (Reddit threads + Google reviews) so the
system sees more than one writing style. Documents marked **REAL** were collected
from live r/ASU threads; the remainder are realistic student-style stand-ins used to
cover buckets the live threads didn't, and are clearly separated so the corpus
provenance is honest.

| # | Source | Type | File path |
|---|--------|------|-----------|
| 1 | r/ASU (**REAL**) | Reddit thread | `documents/reddit_free_parking_hacks_REAL.txt` |
| 2 | r/ASU (**REAL**) | Reddit thread | `documents/reddit_getting_around_transit_bikes_REAL.txt` |
| 3 | r/ASU | Reddit thread | `documents/reddit_permits_cost_waitlist.txt` |
| 4 | r/ASU | Reddit thread | `documents/reddit_which_structure_fills_first.txt` |
| 5 | r/ASU | Reddit thread | `documents/reddit_lightrail_upass_commute.txt` |
| 6 | r/ASU | Reddit thread | `documents/reddit_intercampus_shuttle.txt` |
| 7 | r/ASU | Reddit thread | `documents/reddit_bikes_scooters_theft.txt` |
| 8 | r/ASU | Reddit thread | `documents/reddit_citations_appeals.txt` |
| 9 | r/ASU | Reddit thread | `documents/reddit_gameday_football_parking.txt` |
| 10 | r/Tempe | Reddit thread | `documents/reddit_lightrail_defense_contrarian.txt` |
| 11 | Google Reviews | Review set | `documents/google_reviews_tyler_st_garage.txt` |
| 12 | Google Reviews | Review set | `documents/google_reviews_rural_road_garage.txt` |
| 13 | Google Reviews | Review set | `documents/google_reviews_apache_structure.txt` |

Subtopic coverage: permits, specific structures, light rail / U-Pass, intercampus
shuttles, bikes/scooters, citations & appeals, game-day parking, and free-parking
workarounds. Docs #5 and #10 deliberately **disagree** about whether the light rail
is worth it, to test whether the system surfaces conflicting opinions.

---

## Chunking Strategy

Implemented in `ingest.py` with `RecursiveCharacterTextSplitter`
(separators preferring blank-line / comment / review boundaries, then sentence, then
word).

**Chunk size:** ~500 characters (~100–125 tokens) target.

**Overlap:** 80 characters (~16%).

**Why these choices fit your documents:** The corpus is not long-form prose — it is
collections of **short, self-contained units**: a single Reddit comment or Google
review is usually 1–4 sentences. The ideal chunk is therefore *one complete opinion*
("Tyler St fills first, be there by 8:15"). 500 chars captures one such opinion
without merging unrelated buckets, and stays well under all-MiniLM-L6-v2's **256-token
input cap** (a chunk of 1,500 chars would be silently truncated at embed time — a
correctness bug, not just a quality one). A smaller 200-char chunk would slice
"Rural fills by 9:30 *but the roof is open at 10*" in half, leaving a misleading
fragment. The 16% overlap bridges facts that straddle a boundary.

**Preprocessing before chunking:** `html.unescape` for entities (`&#39;` → `'`),
strip `<tags>`, remove Reddit boilerplate lines (`Upvote`/`Downvote`/`Reply`/`Award`/
`Share`), standalone vote counts, and `• 1y ago` timestamps; the `Source:/Title:/
Topic:` header is parsed into metadata and a short `[source — title]` context prefix
is prepended to each chunk so a review like "fills up fast" still records *which*
garage it describes.

**Final chunk count:** **67 chunks** across 13 documents (lengths 98–495 chars,
median 350, zero empty chunks) — comfortably above the 50 floor and far below the
2,000 ceiling.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, runs locally,
no API key or rate limit). Embeddings are stored in **ChromaDB** with **cosine
distance** (set explicitly so scores map to the "<0.5 = good" guidance; the L2 default
would give squared distances that don't). Top-k = 4.

**Production tradeoff reflection:** If this served real students and cost weren't a
constraint, I'd weigh:
- **Domain accuracy** — a larger model (`text-embedding-3-large`, `bge-large`) better
  separates near-identical complaints ("Tyler fills at 8:15" vs "Rural at 9:30") that
  MiniLM can blur. This is the single biggest quality lever for this corpus, and (as
  the failure analysis shows) recall is my actual bottleneck.
- **Context length** — a longer input window would let me embed full long guides
  without truncation, relaxing the chunking constraints.
- **Multilingual support** — ASU has a large international population; a multilingual
  model (`multilingual-e5`) would let students query in their first language.
- **Latency & local vs. API** — local MiniLM has zero per-query cost and no network
  dependency (great for a free student tool) and keeps student queries on-device for
  privacy; an API model adds latency, cost, and a data-sharing consideration. For
  production I'd likely keep embeddings local but A/B test a hosted large model on the
  eval set before committing.

---

## Grounded Generation

LLM: **Groq `llama-3.3-70b-versatile`**, `temperature=0`. Implemented in `query.py`.

**System prompt grounding instruction:** The model is told (not asked) to use *only*
the supplied passages:

> "Use ONLY facts contained in the numbered context passages. Do NOT use any outside
> or general knowledge, even if you are confident it is correct. If the context does
> not contain enough information to answer the question, reply with exactly this
> sentence and nothing else: *'I don't have enough information on that.'* … When
> sources disagree, say so and present both sides."

Structural reinforcements beyond the prompt: each retrieved chunk is passed as a
**numbered, source-labelled passage** `[1] (source: …)`; `temperature=0` reduces drift
from the provided text; and the strict refusal string is matched in code so the UI can
show "(none)" for sources when the model declines.

**How source attribution is surfaced in the response:** Attribution is **programmatic,
not left to the LLM**. `ask()` returns the de-duplicated list of source filenames for
the retrieved chunks, so a citation exists even if the model forgets to add one. The
model *also* cites passages inline by number (`[1]`, `[2]`). The Gradio UI shows the
answer, the source file list, and an expandable panel with the actual retrieved chunks
and their distance scores, so a viewer can verify grounding directly.

> **Honest nuance:** the source list reflects every *retrieved* document, not only the
> ones the model cited inline. Occasionally a rank-4 chunk from a different file is
> listed as a source even though the answer didn't draw on it. This is a deliberate
> "here's everything the answer could have used" choice, but it means *retrieved* ≠
> *cited*.

---

## Evaluation Report

Run via `python query.py` (and reproducible end-to-end). Distances are cosine
(0 = identical, lower = better).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Which structure fills up earliest, and a workaround for late morning? | Tyler St fills first (~8:15am); Rural Rd **roof** open ~10am; Apache fills latest (~10:30–11). | Tyler first (8:15); Apache open till 10:30–11; Rural roof ~10am; also Novus. All 4 chunks from the right doc (d 0.318–0.353). | Relevant | **Accurate** |
| 2 | Free / cheap ways to park without a permit? | Park-and-ride + light rail; Tempe Beach Park lot; Library + free Orbit; LDS $5 pass; District visitor pass; Lemon/Orange apartments; Speech & Hearing / clinic sign-in. | Library+Orbit, Tempe Beach Park, Lemon/Orange guest parking, Speech & Hearing. **Missed the LDS $5 pass and District visitor pass.** (d 0.316–0.377) | Partially relevant | **Partially accurate** |
| 3 | Can I appeal a ticket if the plate reader misread my plate, and how? | Yes — appeal online with screenshot of permit + payment; ~10–14 day window; "5 minutes" excuses denied. | Yes; online portal + permit/payment screenshot; 10–14 day window; valid-doc appeals succeed, weak excuses denied. (d 0.261–0.499) | Relevant | **Accurate** |
| 4 | Is the U-Pass worth it from Mesa, and how reliable is the light rail? | ~$200/sem unlimited (vs $720 permit); ~30–35 min; every ~12 min daytime; reliable but delays possible; sketchy late at night; one line. | Reliability + delay risk + night-safety + schedule flexibility covered well. **Missed the headline $200-vs-$720 cost case and travel time.** (d 0.187–0.290) | Partially relevant | **Partially accurate** |
| 5 | Overall, should I rely on the light rail instead of driving? | Balanced: pro (cheap, no parking hassle, productive) + con (one line, ~35 min vs 12-min drive, sketchy at night, useless for Poly, bad for gap schedules). | Normal schedule → reliable, $200, productive; non-traditional schedule → car better. **Omitted night-safety, the one-line limit, and the Poly problem.** (d 0.301–0.359) | Partially relevant | **Partially accurate** |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

**Out-of-domain check (not one of the 5):** "What time does the dining hall close?" →
*"I don't have enough information on that."* with no sources — the grounding/refusal
path works as intended.

---

## Failure Case Analysis

**Question that failed:** Q5 — *"Overall, should I rely on the light rail instead of
driving to ASU?"* (Q4 fails the same way and is included as corroborating evidence.)

**What the system returned:** A reasonable but **one-sided** answer: light rail is
reliable, cheap ($200/sem), and lets you be productive on a normal schedule, with the
single caveat that a car is better for irregular schedules. It **omitted** three
caveats that are present in the corpus: the late-night safety concern, the "there's
only one line so you must live near a station" limitation, and the fact that transit
to the Polytechnic campus is impractical.

**Root cause (tied to a specific pipeline stage):** This is a **retrieval-recall
failure at the top-k step, not a generation failure.** With `k=4`, retrieval returned
the 4 chunks most semantically similar to the query — and 3 of them came from the
single document literally framed as "should I use the light rail" (`…defense_
contrarian.txt`). The omitted caveats live in *differently-framed* documents
(night-safety in `…lightrail_upass_commute.txt`, the one-line + Poly limits in
`…getting_around_transit…REAL` and `…intercampus_shuttle.txt`). Those chunks are
genuinely relevant but rank 5th+, so they never enter the top-4 context window and the
LLM literally cannot mention what it never received. Q4 fails identically: the chunk
holding the "$200 vs $720" cost comparison ranked outside the top-4, so the model
answered the reliability half of the question well but the "is it worth it" half
thinly. The embeddings and grounding are working correctly — the bottleneck is that a
*complete* answer to a broad/multi-part question requires more evidence than the 4
closest chunks, especially when relevant facts are spread across several documents.

**What you would change to fix it:** (1) **Raise top-k** to 6–8 for broad/comparative
questions — the cheapest fix, at the cost of more context tokens. (2) Add
**MMR / max-marginal-relevance** retrieval so results are diversified across documents
instead of clustering in the one most-similar doc. (3) Add **hybrid (BM25 + semantic)
search** so a keyword like "Polytechnic" or "$720" pulls its chunk even when semantic
similarity buries it. (4) Longer-term, a stronger embedding model with better
domain separation would lift recall on near-duplicate opinion text.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking Strategy
section in `planning.md` *before* coding forced me to reason about the 256-token cap of
all-MiniLM-L6-v2 and the short, opinion-shaped structure of the documents. That meant
the very first implementation already used boundary-aware ~500-char chunks rather than
a naive fixed split, and it gave me concrete numbers to validate against (the 50–2,000
chunk-count guardrail). When I ran `ingest.py` and saw 67 chunks, I already knew that
was the healthy range — the spec turned "does this look right?" into a checkable
prediction.

**One way your implementation diverged from the spec, and why:** The spec's chunking
plan didn't anticipate **fragment chunks**. On the first ingest run, the recursive
splitter stranded a Reddit username (`Longjumping-Pass2825:`) as its own 21-character
chunk — useless on its own. Reading the printed sample chunks (the inspection step)
surfaced it, so I added two things not in the original plan: a `MIN_CHUNK_CHARS = 80`
merge pass that folds tiny fragments into a neighbor, and a `[source — title]` context
prefix on every chunk so single reviews stay self-contained. I documented both back
into `planning.md`. The divergence came directly from *looking at the data*, which is
exactly why the spec mandates chunk inspection before embedding.

---

## AI Usage

**Instance 1 — Ingestion & chunking implementation**

- *What I gave the AI:* My `planning.md` Documents and Chunking Strategy sections (file
  format, ~500 char / 80 overlap, boundary-aware) and the pipeline diagram, and asked
  it to implement load → clean → chunk.
- *What it produced:* A working `ingest.py` using `RecursiveCharacterTextSplitter` plus
  HTML/boilerplate cleaning and a chunk-inspection printout.
- *What I changed or overrode:* After running it I saw a 21-char username fragment in
  the output, so I directed it to add a small-fragment merge (`MIN_CHUNK_CHARS`) and a
  per-chunk source/context prefix — neither was in the generated first draft. I also
  had it save `chunks.json` in a fixed schema so the next stage could consume it.

**Instance 2 — Vector store distance metric**

- *What I gave the AI:* My Retrieval Approach section and asked it to embed with MiniLM
  and store in ChromaDB with source metadata.
- *What it produced:* Embedding + `retrieve()` code using ChromaDB's default settings.
- *What I changed or overrode:* The default Chroma space is L2, which would have made
  the distance scores incomparable to the spec's "<0.5 = good" threshold. I overrode it
  to **cosine** (`hnsw:space: cosine`) with normalized embeddings so the printed
  distances are interpretable, and verified the change by re-running the test queries
  and confirming top results landed in the 0.18–0.35 range.

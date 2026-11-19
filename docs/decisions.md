# Design decisions

Each decision records the context, the options considered, what was chosen and why, with
the evidence where it was measured. Decisions are numbered in the order they were made.
Items in `[square brackets]` are still to be filled in.

| # | Decision | Date |
|---|---|---|
| D1 | [Models and quantization levels under test](#d1-models-and-quantization-levels-under-test) | 13 Nov 2026 |
| D2 | [CPU-only inference, enforced and checked](#d2-cpu-only-inference-enforced-and-checked) | 15–18 Nov 2026 |
| D3 | [Cloud services only for evaluation](#d3-cloud-services-only-for-evaluation) | 13 Nov 2026 |
| D4 | [Two test machines](#d4-two-test-machines) | 13 Nov 2026 |
| D5 | [Document collection](#d5-document-collection) | 18 Nov 2026 |
| D6 | [Embedding model served by Ollama](#d6-embedding-model-served-by-ollama) | 18 Nov 2026 |
| D7 | [Text cleaning keeps line structure](#d7-text-cleaning-keeps-line-structure) | 18 Nov 2026 |
| D8 | [Chunking: size, overlap and page boundaries](#d8-chunking-size-overlap-and-page-boundaries) | 18–19 Nov 2026 |
| D9 | [Vector store: ChromaDB, offline, idempotent](#d9-vector-store-chromadb-offline-idempotent) | 18 Nov 2026 |
| D10 | [Connect to Ollama at 127.0.0.1, not localhost](#d10-connect-to-ollama-at-127001-not-localhost) | 18 Nov 2026 |
| D11 | [All settings in validated config files](#d11-all-settings-in-validated-config-files) | 18 Nov 2026 |

---

## D1. Models and quantization levels under test

**Context.** The proposal named Llama 3 8B and Phi-3 in one place and Llama 3.2 3B and
Phi-4-mini in another. An 8B model leaves little room in 8 GB of RAM.

**Options.** Older proposal models; current small models (2024–2026) of 3–4B parameters;
larger 7–9B models.

**Decision.** Three models, each at two quantization levels (6 configurations per machine):

| Model | Released | Licence | Role |
|---|---|---|---|
| Llama 3.2 3B | Sep 2024 | Llama 3.2 Community | Established baseline |
| Phi-4-mini (3.8B) | Feb 2025 | MIT | Strong reasoning for its size |
| Qwen3.5 4B | Mar 2026 | Apache 2.0 | Current generation |

Quantization: `Q4_K_M` (4-bit) and `Q8_0` (8-bit) GGUF builds. Qwen3.5's thinking mode is
switched off for benchmarks so its responses are comparable with the other two.

**Reasons.** All three fit in 8 GB at Q4 (2.39–2.93 GiB loaded, measured in the proof of
concept), span two years of model development, and are permissively licensed.

**Set aside.** Gemma 4 E4B: about 5 GB at Q4 (8B total parameters), too large for the laptop.
Optional extras if time allows: Granite 4.0 Micro (RAG-focused), Llama 3.2 1B.

## D2. CPU-only inference, enforced and checked

**Context.** The research question is about hardware without a capable GPU. The desktop has a
GTX 1050 Ti, and Ollama uses an available GPU by default. When an embedding request was sent
without a GPU setting, Ollama tried the GPU and crashed with a CUDA toolchain error.

**Decision.**
- Every Ollama request sends `num_gpu: 0`, set once in `configs/default.yaml` (`ollama.num_gpu`)
  and used for both embeddings and generation.
- On the desktop, the GPU is also hidden from Ollama with the user environment variable
  `CUDA_VISIBLE_DEVICES=-1`.
- The proof-of-concept script checks each run with Ollama's `/api/ps` (`size_vram = 0`) and
  reports PASS or FAIL.

**Evidence.** Every proof-of-concept run on both machines passed the check (0% of the model
on the GPU). Task Manager showed the GPU idle during runs.

## D3. Cloud services only for evaluation

**Context.** The proposal excludes commercial LLM APIs, but automated accuracy scoring
(RAGAS) needs a judge model, and a cloud model is a useful comparison baseline.

**Decision.** The pipeline under test (ingestion, embeddings, vector store, generation) runs
entirely locally and needs no network. Two exceptions, for evaluation only:
1. a cloud judge model to compute RAGAS scores, and
2. a cloud LLM as a comparison baseline.

**Safeguards.** Only public documents are ever sent. The judge is a different model from those
being compared, since a model can favour its own answers. The judge's version and temperature
are fixed and recorded. These exceptions are stated openly in the presentation and report.

## D4. Two test machines

| Machine | CPU | RAM | GPU (disabled) | OS |
|---|---|---|---|---|
| Desktop | AMD Ryzen 5 3600, 6 cores / 12 threads | 16 GB DDR4-2400 | GTX 1050 Ti 4 GB | Windows 10 |
| Laptop | AMD Ryzen 3 5425U, 4 cores / 8 threads | 8 GB DDR4-3200 | Integrated Radeon | Windows 11 |

**Reasons.** Two common consumer tiers. The 8 GB laptop tests where "consumer hardware"
stops being enough. Both run the same code, the same Ollama version and the same document.
Benchmarks run on mains power with background apps closed.

## D5. Document collection

**Context.** The proposal called for domain-specific IT documentation. The documents must be
public and usable for non-commercial research (see D3), and contain specific detail the
models are unlikely to know already, so that retrieval can be shown to matter.

**Decision.** The official *Ubuntu Server documentation* PDF.

| Item | Value |
|---|---|
| Source | https://documentation.ubuntu.com/server ("Current PDF") |
| File | `data/raw/ubuntu-server-guide.pdf` (not committed; git-ignored) |
| Downloaded | '18th Nov 2026' |
| Licence | None stated for the guide as a whole (the repository's licence field is unset); © Canonical Ltd. Website terms allow personal, educational and non-commercial use. One page (p. 994) is CC BY-SA 4.0. |
| SHA-256 | `52C5F63F118F347130E1FDD7564586308EA5E7354CED2088B77404EFC839C7CC` |
| Size | 1,140 pages, 0 empty, all text-based (no OCR needed) |

**Reproducibility.** The download link always serves the latest version, so the same file
(checked by hash) is copied to each machine rather than downloaded again.

**Planned.** A closed-book baseline (the same questions with no retrieved context) in the
main study, to measure how much of each answer comes from the documents rather than from
the model's training.

## D6. Embedding model served by Ollama

**Context.** The proposal names all-MiniLM-L6-v2. It can run through the `sentence-transformers`
Python library or through Ollama (`all-minilm`).

**Decision.** Ollama `all-minilm`.

**Reasons.**
- One local runtime for all models, with the same CPU-only control (D2).
- Avoids installing PyTorch (several GB), which matters on the 8 GB laptop.
- Same model as the proposal, so the method is unchanged.

**Model details** (`ollama show all-minilm`): 23M parameters, F16 (not quantized), embedding
length 384, model context 512 tokens, **default `num_ctx` 256 tokens**. The 256-token limit
sets the chunk size (D8). Only the language models are quantized, so the quantization
comparison concerns generation only.

## D7. Text cleaning keeps line structure

**Context.** The first cleaning step turned single line breaks into spaces. A sample page
showed this flattened a Gitolite config file into one unreadable line. The guide is full of
commands and config files where lines and indentation carry meaning.

**Decision.** Keep line breaks and indentation. Only join lowercase words hyphenated across a
line, collapse repeated spaces inside lines, remove trailing spaces and limit blank lines.

**Reasons.** Line breaks in prose have little effect on embeddings or answers, while lost
structure in code makes answers that quote configuration wrong or unreadable.

## D8. Chunking: size, overlap and page boundaries

**Context.** all-minilm reads at most 256 tokens per text (D6). Text beyond that is cut off and
does not contribute to the chunk's embedding. Technical text (commands, paths, config) uses
more tokens per character than prose.

**Method.** The embedder sends requests with `truncate: false`, so Ollama refuses over-long
chunks instead of silently cutting them. Refused batches are split in half repeatedly until
the long chunks are found; those are embedded with truncation and counted.

**Evidence.**

| Chunk size / overlap (chars) | Sample | Chunks | Tokens per chunk | Truncated |
|---|---|---|---|---|
| 800 / 100 | first 320 chunks | 320 | ~188 | 67 (21%) |
| 600 / 80 | first 320 chunks | 320 | ~140 | 5 (1.6%) |
| 600 / 80 | **full document** | 4,177 | ~146 | **166 (4.0%)** |
| 500 / 70 | **full document** | 4,973 | ~124 | **85 (1.7%)** |

The 320-chunk sample under-estimated truncation: the early sections are mostly prose, the
later reference sections mostly code. Parameters are therefore checked on the full document.

**Decision.** `chunk_size: 500` characters, `chunk_overlap: 70`, with 1.7% of chunks
truncated. This halves truncation compared with 600 / 80, at the cost of 19% more chunks
and slightly less text per retrieved chunk (mean 418 characters). The remaining 1.7% is
accepted as a known limitation; truncated chunks are still stored in full and can be
retrieved, from their first ~256 tokens. A rebuild from scratch gave identical counts
(4,973 chunks, 619,198 tokens, 85 truncated).

**Other choices.**
- Splitting with LangChain's `RecursiveCharacterTextSplitter`, at blank lines first, then line
  breaks, sentence ends and spaces, so chunks rarely break mid-sentence or mid-line.
- Chunks never cross a page boundary, so every chunk cites one exact page. A paragraph running
  over a page break becomes two chunks.
- Fragments under 20 characters (e.g. a lone page number) are dropped.
- Stable chunk IDs: `<file>:p<page>:c<index>`.

## D9. Vector store: ChromaDB, offline, idempotent

**Decision.** ChromaDB, stored on disk in `chroma_db/` (git-ignored, rebuilt with
`python scripts/ingest.py --reset`).
- Cosine distance, the standard measure for sentence embeddings.
- **Offline:** anonymous telemetry switched off; the collection has no built-in embedding
  function, so ChromaDB never downloads or runs a model of its own.
- **Idempotent:** chunks are written with `upsert` on their stable IDs, so re-running
  ingestion overwrites rather than duplicates.
- The collection records which embedding model built it and refuses vectors from another.

**Evidence.** Full ingestion at 500 / 70 (D8) on the desktop: 4,973 chunks embedded in
166.6 s (29.9 chunks/s), 204.3 s in total. Laptop: [ ] chunks in [ ] s ([ ] chunks/s),
[ ] s in total. (At 600 / 80 the desktop took 204.5 s to embed 4,177 chunks, 20.4 chunks/s:
fewer over-long chunks means fewer refused batches to split and retry.)

## D10. Connect to Ollama at 127.0.0.1, not localhost

**Context.** The proof of concept showed a fixed ~2 s time to first token for every model on
both machines, even though prompt processing took about 0.04 s. It was recorded as unexplained.
While embedding documents, every request, whatever its size, took about 2.08 s.

**Cause.** On Windows, `localhost` is tried first as IPv6 (`::1`). Ollama listens on IPv4 only,
so each connection waited about 2 s before falling back to `127.0.0.1`.

**Evidence.**

| Measurement | `localhost` | `127.0.0.1` |
|---|---|---|
| One embedding request | 2.08 s | 0.04 s |
| 320 chunks, embedding only (laptop) | 42.2 s | 21.4 s |

**Decision.** `ollama.host: http://127.0.0.1:11434` in the config, and the same default in the
proof-of-concept script. The proof of concept was re-run; see the measurement correction in
`docs/poc-results.md`. Generation and prefill speeds come from Ollama's internal timings and
are not affected by the connection delay.

## D11. All settings in validated config files

**Decision.** Every setting lives in `configs/*.yaml`, loaded by `local_rag_pipeline.config`,
which rejects unknown or misspelt settings, missing required settings and invalid values
(for example an overlap not smaller than the chunk size), reporting all problems at once.
Each ingestion run saves its summary with a full copy of its settings in `results/ingest/`.

**Reasons.** An experiment is fully described by its config file, and a typo fails
immediately instead of silently falling back to a default during a long benchmark run.

---

## Known limitations

- **Headings lose their spaces** in extraction (e.g. "AddingaGitoliteuser"), from how text is
  stored in this PDF. Body text is unaffected. To revisit with pypdf's layout extraction mode.
- **Page footers** ("9 of 1139") appear in chunks. A one-line filter is planned.
- **Hyphen joining** can occasionally merge a real hyphenated word split at a line end.
- **Chunks don't cross pages**, so content split by a page break is in two chunks.
- **Truncation:** 1.7% of chunks (85 of 4,973) exceed the embedding context at 500
  characters (D8).

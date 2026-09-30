# Agentic Code Intelligence — PRISM Y2026 GenAI Hackathon (Theme 01)

**Team**: **Thapar_Bandits**

**Members**: Lakshya Garg , Goutam Jain , Vaibhav Manchanda , Namish Mahajan

**Demo video:** `https://drive.google.com/file/d/19QdyG7ImKsjEvIOXu9b6YHmmXBtBOW91/view?usp=sharing`

## Problem

Voice-assistant codebases can span thousands of files, and neither a new developer
nor an LLM can hold the whole thing in their head at once. Given a plain-English
question, this project retrieves the most relevant code snippets from a codebase,
along with their file and line locations — without needing to read the entire
repo into a single context window.

## Approach

1. **Baseline embedding retrieval.** Code and queries are both turned into vectors
   using a pretrained sentence embedding model, and ranked by cosine similarity.
2. **Query vs. document preprocessing kept separate.** The pipeline treats natural-
   language questions and code snippets through independent preprocessing steps,
   since useful transformations differ for each (e.g. identifier-splitting only
   makes sense on code).
3. **Chunking with overlap.** For the live demo, files are cut into overlapping
   30-line windows (15-line stride) rather than embedded whole, so a function
   split across a chunk boundary is still captured intact somewhere in the index.

## Tech stack

- Python, [sentence-transformers](https://www.sbert.net/) (`BAAI/bge-small-en-v1.5`)
- [MTEB](https://github.com/embeddings-benchmark/mteb) for standardized evaluation
- NumPy for similarity search (cosine similarity via normalized dot product)
- CPU-only, no GPU required

## Results

Evaluated on the CoIR **AppsRetrieval** test split (3,765 queries, 8,765 code
snippets), via MTEB:

| Metric | Score |
|---|---|
| NDCG@10 | 0.0514 |
| MRR@10 | 0.0437 |

Full per-metric breakdown: [`appsretrieval_results.json`](./appsretrieval_results.json).

For context, code-specialized embedding models (e.g. trained specifically on
source code) report NDCG@10 in the 40-50 range on this same task in published
benchmarks. This baseline uses a general-purpose text embedding model, so the
gap is expected — see **Limitations** below.

## Limitations & future work

- **General-purpose embedding model, not code-trained.** We attempted to swap in
  a code-specialized model (`Salesforce/SFR-Embedding-Code-400M_R`), but it
  failed on our setup with a compatibility error in its custom model code. This
  is the highest-leverage next step for improving retrieval accuracy.
- **Identifier splitting (e.g. `preProcessInput` -> `pre Process Input`) was
  implemented but not fully benchmarked** - the run hit a memory limit on our
  hardware before completing. The preprocessing hook (`_preprocess_document`)
  is already in place and can be toggled on (`USE_IDENTIFIER_SPLIT`) once
  tested on a machine with more available memory.
- **No structural/call-graph retrieval yet.** Queries like "which files call X
  before Y" require AST or call-graph indexing, which is out of scope for this
  submission but is a natural extension - see `demo.py`'s chunking approach,
  which would be replaced by function/AST-level chunks.
- **Fixed-window chunking, not AST-aware.** The live demo currently chunks by
  line windows rather than by function boundary.

## How to run

### 1. Install dependencies
```bash
pip install mteb sentence-transformers numpy
```

### 2. Reproduce the evaluation score
```bash
python baselineencoder.py
```
Downloads the CoIR AppsRetrieval dataset and model on first run (subsequent
runs use the local cache). Prints NDCG@10 / MRR@10 and writes
`appsretrieval_results.json`.

### 3. Run the live demo
```bash
python demo.py path/to/a/js/codebase
```
Indexes the given folder (skips `node_modules`, `.git`, `dist`, `build`), then
prompts for natural-language questions and returns the top-matching code
snippets with file, line range, and similarity score. Add `--rebuild` to
re-index after the codebase changes, and `--k N` to control how many results
are shown.

## Repo contents

| File | Purpose |
|---|---|
| `baselineencoder.py` | MTEB-compatible encoder; produces the official evaluation score |
| `demo.py` | Interactive search over a real codebase — the working prototype |
| `appsretrieval_results.json` | Official MTEB evaluation output (required submission artifact) |

## Constraints satisfied

- Runs on CPU only, no GPU dependency
- Output is snippets + file/line locations (no full code generation)
- Query preprocessing and code preprocessing are separated, ready for independent tuning

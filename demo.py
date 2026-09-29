import argparse
import os
import pickle
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TRUST_REMOTE_CODE = False
MAX_SEQ_LENGTH = 512
CODE_EXTENSIONS = (".js", ".jsx", ".mjs", ".ts", ".tsx", ".py")
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".next"}
CHUNK_LINES = 30
CHUNK_STRIDE = 15

def split_identifiers(text: str) -> str:
    """preProcessInput -> pre Process Input, my_var_name -> my var name."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text.replace("_", " ")

def find_code_files(root: str) -> list[str]:
    paths = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(CODE_EXTENSIONS):
                paths.append(os.path.join(folder, name))
    return sorted(paths)

def chunk_file(path: str) -> list[dict]:
    """Cut one file into overlapping windows of lines. Line numbers start at 1."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    except OSError:
        return []

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        text = "\n".join(lines[start:end])
        if text.strip():
            chunks.append(
                {"file": path, "start": start + 1, "end": end, "text": text}
            )
        if end == len(lines):
            break
        start += CHUNK_STRIDE
    return chunks

def build_index(model, root: str, cache_path: str) -> tuple[list[dict], np.ndarray]:
    t0 = time.perf_counter()
    chunks = []
    for path in find_code_files(root):
        chunks.extend(chunk_file(path))
    if not chunks:
        raise SystemExit(f"No code files found under: {root}")

    texts = [split_identifiers(c["text"]) for c in chunks]
    vectors = model.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    with open(cache_path, "wb") as f:
        pickle.dump((chunks, vectors), f)

    print(f"Indexed {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")
    return chunks, vectors

def search(model, chunks, vectors, question: str, k: int):
    t0 = time.perf_counter()
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = vectors @ q
    best = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in best], time.perf_counter() - t0

def show(results, elapsed: float) -> None:
    print(f"\nTop {len(results)} results  ({elapsed * 1000:.0f} ms)")
    for rank, (c, score) in enumerate(results, 1):
        print(f"\n#{rank}  {c['file']}  lines {c['start']}-{c['end']}   score {score:.3f}")
        preview = c["text"].splitlines()[:8]
        for line in preview:
            print("    " + line[:110])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder containing the code to search")
    parser.add_argument("--k", type=int, default=5, help="results to show")
    parser.add_argument("--rebuild", action="store_true", help="ignore the saved index")
    args = parser.parse_args()
    model = SentenceTransformer(MODEL_NAME, device="cpu", trust_remote_code=TRUST_REMOTE_CODE)
    model.max_seq_length = MAX_SEQ_LENGTH

    cache_path = os.path.join(args.folder, ".code_index.pkl")
    if os.path.exists(cache_path) and not args.rebuild:
        with open(cache_path, "rb") as f:
            chunks, vectors = pickle.load(f)
        print(f"Loaded saved index ({len(chunks)} chunks)")
    else:
        chunks, vectors = build_index(model, args.folder, cache_path)

    print("\nAsk a question about the code (empty line to quit).")
    while True:
        question = input("\nAsk> ").strip()
        if not question:
            break
        results, elapsed = search(model, chunks, vectors, question, args.k)
        show(results, elapsed)
if __name__ == "__main__":
    main()
<!-- Filename: 2025-09-06_til_sqlite-vec-openai-embeddings.md -->

# Add Vector Search to SQLite with sqlite-vec and OpenAI

## What
Use `sqlite-vec` to store/query embeddings directly in SQLite, generating vectors with OpenAI’s `text-embedding-3-small` (1536-dim) model.

## Context
- Stack: Python 3.11+, `sqlite3`, `sqlite-vec`, `openai`.
- Requires: `OPENAI_API_KEY` in environment; sqlite loadable extensions enabled.
- Suitable for: lightweight RAG or semantic search without a separate vector DB.

## Steps / Snippet
```bash
# 1) Install dependencies in your project environment
poetry add openai sqlite-vec

# 2) Export your OpenAI API key
export OPENAI_API_KEY=YOUR_KEY

# 3) Run your script (example path)
poetry run python -m src.scripts.sqlite_vector_extension
```

```python
# minimal example: create table, insert OpenAI embeddings, query top-k
import os, sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
from openai import OpenAI

MODEL = "text-embedding-3-small"
DIMS = 1536

client = OpenAI()  # uses OPENAI_API_KEY

con = sqlite3.connect("/path/to/your.db")
con.enable_load_extension(True)
sqlite_vec.load(con)
con.enable_load_extension(False)

con.execute(
    f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{DIMS}], content TEXT)"
)

texts = [
    "The sky is blue because of Rayleigh scattering.",
    "Artificial intelligence is intelligence demonstrated by machines.",
]
embs = client.embeddings.create(model=MODEL, input=texts)

with con:
    for t, d in zip(texts, embs.data):
        con.execute(
            "INSERT INTO vec_items(embedding, content) VALUES (?, ?)",
            (serialize_float32(d.embedding), t),
        )

# Query with top-k (lower distance = more similar)
q = "Why is the sky blue?"
q_emb = client.embeddings.create(model=MODEL, input=[q]).data[0].embedding
rows = con.execute(
    """
    SELECT content, distance
    FROM vec_items
    WHERE embedding MATCH ?
    AND k = 3
    ORDER BY distance
    """,
    [serialize_float32(q_emb)],
).fetchall()

for i, (content, distance) in enumerate(rows, 1):
    print(f"{i}. {content} (distance: {distance:.4f})")
```

## Pitfalls
- aarch64 wheel mismatch: On some ARM64 environments, you may see `wrong ELF class: ELFCLASS32` when loading `sqlite-vec`. Fix by installing a native build (e.g., `poetry run pip install --no-binary=:all: --no-cache-dir sqlite-vec`), or install from the project’s `bindings/python` after cloning, or manually load a known-good `vec0.so` via `conn.load_extension(...)`.

## Links
- interactive demo: https://gist.github.com/LxYuan0420/010945c08ac74b862f65c9ec891c5e50
- sqlite-vec (GitHub): https://github.com/asg017/sqlite-vec
- OpenAI embeddings: https://platform.openai.com/docs/guides/embeddings

"""Generate normalized SPECTER2 base embeddings in Parquet row order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from embedding_model import (
    DIMENSION,
    MODEL_ID,
    Specter2Encoder,
    format_paper,
    paper_ids,
    sha256_file,
    validate_vectors,
)

OUTPUT_NAMES = ("embeddings.npy", "source_ids.json", "encoded_ids.json", "metadata.json")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/arxiv_subset.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--quiet", action="store_true", help="omit the progress bar")
    parser.add_argument("--overwrite", action="store_true", help="replace previous embedding artifacts")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.overwrite and any((args.output_dir / name).exists() for name in OUTPUT_NAMES):
        parser.error("embedding artifacts already exist; use another --output-dir or --overwrite")

    start = time.perf_counter()
    parquet_hash = sha256_file(args.input)
    frame = pd.read_parquet(args.input, columns=["id", "title", "abstract"])
    source_ids = paper_ids(frame)
    items = []
    for identifier, title, abstract in frame.itertuples(index=False, name=None):
        if any(not isinstance(value, str) or not value.strip() for value in (title, abstract)):
            raise ValueError(f"Missing title or abstract for article {identifier}.")
        items.append((identifier, format_paper(title, abstract)))

    encoder = Specter2Encoder(device=args.device)
    print(f"Model: {MODEL_ID}@{encoder.revision}; CLS; no adapter; device={encoder.device}", flush=True)
    vectorize_start = time.perf_counter()
    vectors = encoder.encode(
        [text for _, text in items], batch_size=args.batch_size, show_progress=not args.quiet
    )
    vectorize_seconds = time.perf_counter() - vectorize_start
    encoded_ids = [identifier for identifier, _ in items]
    if encoded_ids != source_ids:
        raise RuntimeError("Encoded ID order does not match source IDs.")
    norms = validate_vectors(vectors, len(encoded_ids))
    if sha256_file(args.input) != parquet_hash:
        raise RuntimeError("Parquet changed during embedding; no artifacts were written.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=args.output_dir) as directory:
        stage = Path(directory)
        np.save(stage / "embeddings.npy", vectors, allow_pickle=False)
        write_json(stage / "source_ids.json", source_ids)
        write_json(stage / "encoded_ids.json", encoded_ids)
        write_json(stage / "metadata.json", {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_id": MODEL_ID,
            "model_revision": encoder.revision,
            "adapter": None,
            "pooling": "cls",
            "normalize_embeddings": True,
            "max_seq_length": encoder.max_seq_length,
            "text_format": "title [SEP] abstract",
            "dimension": DIMENSION,
            "dtype": str(vectors.dtype),
            "rows": len(encoded_ids),
            "device": encoder.device,
            "batch_size": args.batch_size,
            "parquet_file": args.input.name,
            "parquet_sha256": parquet_hash,
            "embeddings_sha256": sha256_file(stage / "embeddings.npy"),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "vectorize_seconds": round(vectorize_seconds, 3),
            "packages": {name: version(name) for name in (
                "numpy", "torch", "transformers", "sentence-transformers", "pandas", "pyarrow"
            )},
        })
        # The manifest is the final file replaced. Consumers fail closed on incomplete sets.
        for name in OUTPUT_NAMES:
            target = args.output_dir / name
            if not args.overwrite and target.exists():
                raise RuntimeError(f"Output already exists: {target}")
            os.replace(stage / name, target)

    print(f"Оброблено текстів: {len(encoded_ids)}")
    print(f"Розмірність: {vectors.shape[1]}")
    print(f"Норма першого ембеддингу: {norms[0]:.8f}")
    print(f"Діапазон норм: {norms.min():.8f}–{norms.max():.8f}")
    print(f"Час векторизації: {vectorize_seconds:.2f} с; загальний час: {time.perf_counter() - start:.2f} с")
    print(f"Вихід: {args.output_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()

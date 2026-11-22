"""Shared SPECTER2 representation and verified corpus-vector loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

MODEL_ID = "allenai/specter2_base"
MODEL_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
DIMENSION = 768
MAX_SEQ_LENGTH = 512


def format_paper(title: str, abstract: str) -> str:
    return f"{title.strip()} [SEP] {abstract.strip()}"


class Specter2Encoder:
    """Use the rubric's base model with explicit first-token pooling."""

    max_seq_length = MAX_SEQ_LENGTH

    def __init__(self, device: str = "auto", revision: str = MODEL_REVISION):
        import torch
        from sentence_transformers import SentenceTransformer, models

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable. Use --device cpu.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu.")
        arguments = {"revision": revision, "trust_remote_code": False}
        transformer = models.Transformer(
            MODEL_ID,
            max_seq_length=self.max_seq_length,
            # The pinned Allen AI revision publishes pytorch_model.bin.
            # Avoid attempting an unpinned remote safetensors conversion.
            model_args={**arguments, "use_safetensors": False},
            tokenizer_args=arguments.copy(),
            config_args=arguments.copy(),
        )
        if transformer.get_word_embedding_dimension() != DIMENSION:
            raise ValueError("Unexpected hidden size for SPECTER2 base.")
        pooling = models.Pooling(
            DIMENSION, pooling_mode_cls_token=True, pooling_mode_mean_tokens=False
        )
        self.model = SentenceTransformer(modules=[transformer, pooling], device=device)
        self.tokenizer = transformer.tokenizer
        if self.tokenizer.sep_token != "[SEP]":
            raise ValueError("Unexpected SPECTER2 separator.")
        self.device = device
        self.revision = revision

    def encode(
        self, texts: Sequence[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not a single string.")
        items = list(texts)
        if batch_size < 1 or not items or any(not t.strip() for t in items):
            raise ValueError("At least one non-empty text and a positive batch size are required.")
        vectors = self.model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        validate_vectors(vectors, len(items))
        return vectors


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paper_ids(frame) -> list[str]:
    ids = frame["id"].tolist()
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("Each article must have a non-empty string ID.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate arXiv IDs in Parquet.")
    return ids


def validate_vectors(vectors: np.ndarray, rows: int) -> np.ndarray:
    if rows < 1 or vectors.shape != (rows, DIMENSION):
        raise ValueError(f"Expected shape ({rows}, {DIMENSION}), received {vectors.shape}.")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite float32 vectors.")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
        raise ValueError("All vector rows must have unit L2 norm.")
    return norms


def load_aligned_embeddings(input_path, embeddings_dir):
    """Reject stale or mismatched artifacts before joining vectors to metadata."""
    import pandas as pd

    root = Path(embeddings_dir)
    with (root / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError("Unrecognized embedding manifest schema.")
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pooling": "cls",
        "normalize_embeddings": True,
        "max_seq_length": MAX_SEQ_LENGTH,
        "dimension": DIMENSION,
        "dtype": "float32",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Embedding manifest describes a different encoder configuration.")
    if sha256_file(input_path) != metadata.get("parquet_sha256"):
        raise ValueError("Parquet content changed after embedding; run 02_embed.py again.")
    frame = pd.read_parquet(input_path)
    ids = paper_ids(frame)
    if len(ids) != metadata.get("rows"):
        raise ValueError("Embedding manifest row count differs from Parquet.")
    for name in ("source_ids.json", "encoded_ids.json"):
        with (root / name).open(encoding="utf-8") as handle:
            if json.load(handle) != ids:
                raise ValueError(f"{name}: row order differs from the source Parquet.")
    vector_path = root / "embeddings.npy"
    if sha256_file(vector_path) != metadata.get("embeddings_sha256"):
        raise ValueError("Vector checksum differs from the embedding manifest.")
    matrix = np.load(vector_path, allow_pickle=False)
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Expected one NumPy array in embeddings.npy.")
    validate_vectors(matrix, len(ids))
    return frame, matrix, metadata

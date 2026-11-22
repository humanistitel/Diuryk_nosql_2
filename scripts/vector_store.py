"""Shared paths, aligned paper vectors, and the Pinecone vector API."""

from collections.abc import Iterable
from hashlib import sha256
from itertools import islice
from pathlib import Path
from time import monotonic, sleep
import os

from embedding_model import DIMENSION, load_aligned_embeddings, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data/arxiv_subset.parquet"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings"
PAPER_INDEX = "arxiv-papers"


def paper_vectors(data_path=None, embeddings_dir=None):
    frame, vectors, _ = load_aligned_embeddings(
        Path(data_path or DEFAULT_DATA), Path(embeddings_dir or DEFAULT_EMBEDDINGS)
    )
    return frame.reset_index(drop=True), vectors


def paper_namespace(data_path=None, embeddings_dir=None) -> str:
    digest = sha256()
    for path in (Path(data_path or DEFAULT_DATA), Path(embeddings_dir or DEFAULT_EMBEDDINGS) / "embeddings.npy"):
        digest.update(bytes.fromhex(sha256_file(path)))
    return f"papers-{digest.hexdigest()[:16]}"


def pinecone_index(name: str, *, create: bool = False):
    from dotenv import load_dotenv
    from pinecone import Pinecone, ServerlessSpec

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Set PINECONE_API_KEY in .env or select --backend local.")

    client = Pinecone(api_key=api_key)
    if create and not client.has_index(name):
        client.create_index(
            name=name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
            timeout=120,
        )

    deadline = monotonic() + 120
    while True:
        description = client.describe_index(name)
        if description.dimension != DIMENSION or description.metric != "cosine":
            raise ValueError(f"Index {name} must have dimension {DIMENSION} and cosine metric.")
        if description.status.ready:
            return client.Index(host=description.host)
        if monotonic() >= deadline:
            raise TimeoutError(f"Index {name} did not become ready.")
        sleep(2)


def upsert_records(index, records: Iterable[dict], *, namespace: str, batch_size: int = 64) -> int:
    from tqdm import tqdm

    if not 1 <= batch_size <= 100:
        raise ValueError("Use batches of 1–100 records for this dataset.")
    records = iter(records)
    written = 0
    with tqdm(desc=f"Upsert {namespace}", unit="vectors") as progress:
        while batch := list(islice(records, batch_size)):
            result = index.upsert(vectors=batch, namespace=namespace, _request_timeout=30)
            if result.upserted_count != len(batch):
                raise RuntimeError(f"Pinecone acknowledged {result.upserted_count} of {len(batch)} records.")
            written += len(batch)
            progress.update(len(batch))
    return written


def wait_for_count(index, expected: int, *, namespace: str, timeout: float = 120) -> int:
    deadline = monotonic() + timeout
    while True:
        stats = index.describe_index_stats(_request_timeout=min(timeout, 15))
        info = (stats.namespaces or {}).get(namespace)
        count = info.vector_count if info is not None else 0
        if count == expected:
            return count
        if count > expected:
            raise ValueError(f"Namespace {namespace} contains {count}, expected {expected} vectors.")
        if monotonic() >= deadline:
            raise TimeoutError(f"Namespace {namespace} reports {count}/{expected} indexed vectors.")
        sleep(2)

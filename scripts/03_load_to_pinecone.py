"""Prepare paper records and optionally upsert them to arxiv-papers."""

import argparse
import json
from pathlib import Path

import pandas as pd

from vector_store import (
    DEFAULT_DATA,
    DEFAULT_EMBEDDINGS,
    PAPER_INDEX,
    paper_namespace,
    paper_vectors,
    pinecone_index,
    upsert_records,
    wait_for_count,
)


def paper_records(frame, vectors):
    for i, paper in enumerate(frame.itertuples(index=False)):
        metadata = {
            "arxiv_id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract[:500],
            "authors": paper.authors[:200] if pd.notna(paper.authors) else "",
            "category": paper.category,
        }
        if pd.notna(paper.year):
            metadata["year"] = int(paper.year)
        if len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) > 40 * 1024:
            raise ValueError(f"Paper {paper.id} exceeds the Pinecone metadata size limit.")
        yield {"id": f"paper_{i}", "values": vectors[i].tolist(), "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "pinecone"), default="local")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be in 1–100.")

    frame, vectors = paper_vectors(args.data, args.embeddings_dir)
    namespace = paper_namespace(args.data, args.embeddings_dir)
    print(f"Backend: {args.backend}. Index: {PAPER_INDEX}. Namespace: {namespace}.")
    print(f"Verified alignment: {len(frame)} papers, {vectors.shape[1]} dimensions, {vectors.dtype}.")
    records = paper_records(frame, vectors)

    if args.backend == "pinecone":
        index = pinecone_index(PAPER_INDEX, create=True)
        written = upsert_records(index, records, namespace=namespace, batch_size=args.batch_size)
        count = wait_for_count(index, len(frame), namespace=namespace)
        total = index.describe_index_stats().total_vector_count
        print(f"Acknowledged upserts: {written}. Indexed in namespace: {count}. Whole index: {total}.")
    else:
        count = 0
        for count, record in enumerate(records, start=1):
            if count <= 2:
                metadata = record["metadata"]
                print(f"{record['id']} → {metadata['arxiv_id']}, year={metadata.get('year')}, category={metadata['category']}.")
        print(f"Validated local Pinecone payloads: {count}. Cloud index count: unmeasured.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TimeoutError) as error:
        raise SystemExit(str(error)) from error

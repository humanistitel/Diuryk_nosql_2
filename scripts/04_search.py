"""Search the arXiv corpus with SPECTER2, metadata filters, and three exact metrics."""

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from embedding_model import Specter2Encoder
from vector_store import (
    DEFAULT_DATA,
    DEFAULT_EMBEDDINGS,
    PAPER_INDEX,
    paper_namespace,
    paper_vectors,
    pinecone_index,
    wait_for_count,
)


def stable_top(scores, rows, k, *, descending):
    order = np.lexsort((rows, -scores if descending else scores))[:k]
    return [(int(rows[i]), float(scores[i])) for i in order]


def scenario_masks(frame, as_of_year):
    year = frame["year"]
    lower = as_of_year - 4
    recent = (year.between(lower, as_of_year) & frame["category"].eq("cs.LG")).fillna(False)
    older = year.lt(2015).fillna(False)
    return (
        ("unfiltered", np.ones(len(frame), dtype=bool), None),
        (
            f"cs.LG in {lower}–{as_of_year}",
            recent.to_numpy(dtype=bool),
            {"year": {"$gte": lower, "$lte": as_of_year}, "category": {"$eq": "cs.LG"}},
        ),
        ("before 2015", older.to_numpy(dtype=bool), {"year": {"$lt": 2015}}),
    )


def matched_papers(index, frame, query, mask, cloud_filter, namespace, top_k):
    response = index.query(
        vector=query.tolist(), top_k=top_k, filter=cloud_filter,
        namespace=namespace, include_metadata=True, _request_timeout=15,
    )
    result = []
    for match in response.matches:
        prefix, _, number = match.id.partition("_")
        if prefix != "paper" or not number.isascii() or not number.isdigit() or int(number) >= len(frame):
            raise ValueError(f"Unexpected paper ID returned by Pinecone: {match.id}")
        row = int(number)
        if not mask[row] or (match.metadata or {}).get("arxiv_id") != frame.iloc[row]["id"]:
            raise ValueError(f"Pinecone result {match.id} does not match the local corpus and filter.")
        result.append((row, float(match.score)))
    return result


def describe(frame, results):
    entries = []
    for row, score in results:
        record = frame.iloc[row]
        entries.append({
            "id": str(record["id"]),
            "title": str(record["title"]),
            "year": int(record["year"]) if pd.notna(record["year"]) else None,
            "category": str(record["category"]),
            "abstract_excerpt": str(record["abstract"])[:180],
            "score": score,
        })
    return entries


def print_results(label, entries):
    print(f"\n{label}")
    if not entries:
        print("No matching records.")
    for i, entry in enumerate(entries, start=1):
        print(f"{i}. {entry['score']:.6f} | {entry['id']} | {entry['year']} {entry['category']} | {entry['title']}")
        print(f"   {entry['abstract_excerpt']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "pinecone"), default="local")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--as-of-year", type=int, default=date.today().year)
    parser.add_argument("--query", default="teaching machines to recognize objects in pictures")
    parser.add_argument("--filter-query", default="reinforcement learning")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100 or not 1905 <= args.as_of_year <= 9999:
        parser.error("Use --top-k in 1–100 and --as-of-year in 1905–9999.")

    frame, vectors = paper_vectors(args.data, args.embeddings_dir)
    encoder = Specter2Encoder(device=args.device)
    query, filtered_query = encoder.encode([args.query, args.filter_query], show_progress=False)
    index = pinecone_index(PAPER_INDEX) if args.backend == "pinecone" else None
    namespace = paper_namespace(args.data, args.embeddings_dir)
    if index is not None:
        wait_for_count(index, len(frame), namespace=namespace, timeout=15)
    result = {"backend": args.backend, "namespace": namespace, "cases": {}, "metrics": {}}
    print(f"Backend: {args.backend}. Namespace: {namespace}. Corpus: {len(frame)}.")

    vectors64 = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors64, axis=1)
    for label, mask, cloud_filter in scenario_masks(frame, args.as_of_year):
        q = query if cloud_filter is None else filtered_query
        text = args.query if cloud_filter is None else args.filter_query
        rows = np.flatnonzero(mask)
        if index is None:
            dot = vectors64[rows] @ q.astype(np.float64)
            scores = dot / (norms[rows] * np.linalg.norm(q.astype(np.float64)))
            matches = stable_top(scores, rows, args.top_k, descending=True)
        else:
            matches = matched_papers(index, frame, q, mask, cloud_filter, namespace, args.top_k)
        entries = describe(frame, matches)
        result["cases"][label] = {"query": text, "eligible": len(rows), "matches": entries}
        print_results(f"{label} | {text} | eligible: {len(rows)}", entries)

    q64 = query.astype(np.float64)
    dot = vectors64 @ q64
    cosine = dot / (norms * np.linalg.norm(q64))
    l2 = np.linalg.norm(vectors64 - q64, axis=1)
    for metric, scores, descending in (("cosine", cosine, True), ("dot", dot, True), ("l2", l2, False)):
        entries = describe(frame, stable_top(scores, np.arange(len(frame)), args.top_k, descending=descending))
        result["metrics"][metric] = entries
        print_results(f"Exact local {metric} | {args.query}", entries)

    ranked_ids = [[entry["id"] for entry in entries] for entries in result["metrics"].values()]
    result["same_order"] = ranked_ids[0] == ranked_ids[1] == ranked_ids[2]
    print(f"Identical exact top-{args.top_k} ordering: {result['same_order']}.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, TimeoutError) as error:
        raise SystemExit(str(error)) from error

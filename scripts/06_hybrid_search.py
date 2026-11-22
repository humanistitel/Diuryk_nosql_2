"""Порівняння лексичного BM25, векторного пошуку та їх поєднання через RRF."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
INDEX_NAME = "arxiv-papers"
QUERIES = ("BERT fine-tuning", "Yann LeCun convolutional networks",
           "making computers understand human emotions from text")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(value):
    return WORD.findall(unicodedata.normalize("NFKC", value).casefold())


def lexical_rank(index, occurrences, query, candidate_k):
    terms = tuple(dict.fromkeys(tokenize(query)))
    candidates = sorted(set().union(*(occurrences.get(term, ()) for term in terms)))
    if not candidates:
        return []
    scores = index.get_batch_scores(terms, candidates)
    return sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))[:candidate_k]


def reciprocal_rank_fusion(rankings, k=60):
    """Відсутній ID дає нуль внеску; вихідні значення подібності не додаються."""
    if k < 1:
        raise ValueError("rrf-k має бути додатним")
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, row in enumerate(dict.fromkeys(ranking), start=1):
            scores[row] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def vector_rank_local(vectors, query, candidate_k):
    scores = vectors @ query
    indices = np.argsort(-scores, kind="stable")[:candidate_k]
    return [(int(idx), float(scores[idx])) for idx in indices]


def vector_rank_pinecone(index, namespace, paper_ids, query, candidate_k):
    response = index.query(vector=query.tolist(), namespace=namespace, top_k=candidate_k,
                           include_metadata=True, _request_timeout=30)
    rows = []
    for match in response.matches:
        match_id = re.fullmatch(r"paper_(0|[1-9][0-9]*)", match.id)
        if not match_id:
            raise ValueError(f"Pinecone повернув незнайомий ID: {match.id}")
        row = int(match_id.group(1))
        if row >= len(paper_ids) or (match.metadata or {}).get("arxiv_id") != paper_ids[row]:
            raise ValueError(f"ID {match.id} не належить до узгодженого корпусу")
        rows.append((row, float(match.score)))
    return rows


def show(name, hits, papers, n):
    print(f"  {name}:")
    if not hits:
        print("    Немає збігів.")
    for rank, (idx, score) in enumerate(hits[:n], start=1):
        paper = papers.iloc[idx]
        print(f"    {rank}. {paper['title']} | arXiv:{paper['id']} | {paper['category']} | {score:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/arxiv_subset.parquet")
    parser.add_argument("--embeddings-dir", type=Path, default=ROOT / "embeddings")
    parser.add_argument("--backend", choices=("local", "pinecone"), default="local")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--compare-rrf-k", type=int, default=1)
    parser.add_argument("--query", action="append", help="Замінює типові запити; можна повторити.")
    args = parser.parse_args()
    if min(args.batch_size, args.top_k, args.rrf_k, args.compare_rrf_k) < 1 or args.candidate_k < args.top_k:
        parser.error("Розміри мають бути додатними й потрібно candidate-k ≥ top-k.")
    queries = tuple(q.strip() for q in (args.query or QUERIES))
    if not all(queries):
        parser.error("Запит не може бути порожнім.")

    from embedding_model import Specter2Encoder
    from vector_store import paper_namespace, paper_vectors, pinecone_index, wait_for_count

    papers, embeddings = paper_vectors(args.data, args.embeddings_dir)
    papers = papers.reset_index(drop=True)
    paper_ids = papers["id"].tolist()
    tokenized = [tokenize(" ".join((str(row.title), str(row.abstract), str(row.authors or ""))))
                 for row in papers.fillna({"authors": ""}).itertuples(index=False)]
    occurrences = defaultdict(set)
    for row, words in enumerate(tokenized):
        for word in set(words):
            occurrences[word].add(row)
    if not occurrences:
        raise ValueError("Для BM25 потрібен непорожній корпус із лексичними токенами")
    bm25 = BM25Okapi(tokenized)
    index = None
    namespace = None
    if args.backend == "pinecone":
        namespace = paper_namespace(args.data, args.embeddings_dir)
        index = pinecone_index(INDEX_NAME)
        wait_for_count(index, len(papers), namespace=namespace)
    encoder = Specter2Encoder(device=args.device)
    query_vectors = encoder.encode(queries, batch_size=args.batch_size, show_progress=False)

    print(f"Режим: {args.backend}; статей: {len(papers)}; "
          f"до {args.candidate_k} кандидатів від кожного методу; RRF k={args.rrf_k}.")
    for query, vector in zip(queries, query_vectors):
        lexical = lexical_rank(bm25, occurrences, query, args.candidate_k)
        semantic = (vector_rank_local(embeddings, vector, args.candidate_k) if index is None else
                    vector_rank_pinecone(index, namespace, paper_ids, vector, args.candidate_k))
        rankings = ([row for row, _ in lexical], [row for row, _ in semantic])
        fused = reciprocal_rank_fusion(rankings, args.rrf_k)
        alternate = reciprocal_rank_fusion(rankings, args.compare_rrf_k)
        print(f"\nЗапит: {query}")
        show("BM25", lexical, papers, args.top_k)
        show("Векторний cosine", semantic, papers, args.top_k)
        show(f"Гібридний RRF (k={args.rrf_k})", fused, papers, args.top_k)
        show(f"Порівняння RRF (k={args.compare_rrf_k})", alternate, papers, args.top_k)
        shown_lexical, shown_semantic = (set(rows[:args.top_k]) for rows in rankings)
        new_to_top = [str(papers.iloc[row]["id"]) for row, _ in fused[:args.top_k]
                      if row not in shown_lexical | shown_semantic]
        print(f"  Лише у гібридному top-{args.top_k}: {', '.join(new_to_top) or 'немає'}.")


if __name__ == "__main__":
    main()

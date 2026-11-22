"""Порівняння фрагментів фіксованої довжини та фрагментів за реченнями."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEX_NAMES = ("arxiv-chunks-fixed", "arxiv-chunks-semantic")
QUERIES = (
    "image classification using convolutional neural networks",
    "reward-driven learning and sequential decision making",
    "phase transitions in interacting quantum systems",
)
SENTENCE_END = re.compile(r"[.!?]+[\"'”’»)]*(?=\s+|$)")
ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|et\s+al|cf|fig|eq|sec|ref|vs|dr|mr|mrs|prof|no)\.$", re.I)
INITIAL = re.compile(r"\b(?:[A-Z]\.)+$")
ID_PATTERN = re.compile(r"^\S+$")


def sentences(text):
    """Поділ за пунктуацією з поправкою на поширені наукові скорочення."""
    text = " ".join(text.split())
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        part = text[start:end].strip()
        if end < len(text) and (ABBREVIATION.search(part) or INITIAL.search(part)):
            continue
        if part:
            yield part
        start = end
    if tail := text[start:].strip():
        yield tail


def fixed_chunks(text, max_words, overlap):
    if max_words <= 0 or not 0 <= overlap < max_words:
        raise ValueError("Потрібно 0 ≤ overlap < max-words")
    words = text.split()
    chunks = []
    for start in range(0, len(words), max_words - overlap):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def sentence_chunks(text, max_words):
    """Групує цілі речення; розбиває лише речення, довші за сам ліміт."""
    if max_words <= 0:
        raise ValueError("max-words має бути додатним")
    chunks, current = [], []
    for sentence in sentences(text):
        words = sentence.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        while len(words) > max_words:
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_longest(path, limit):
    columns = ["id", "title", "abstract", "year", "category"]
    frame = pd.read_parquet(path, columns=columns)
    clean = frame.copy()
    for field in ("id", "title", "abstract", "category"):
        clean[field] = clean[field].astype("string").str.strip()
        if clean[field].isna().any() or clean[field].eq("").any():
            raise ValueError(f"Порожні значення в колонці {field}")
    if clean["id"].duplicated().any() or not clean["id"].str.fullmatch(ID_PATTERN).all():
        raise ValueError("Потрібні унікальні arXiv ID без пробілів")
    clean["word_count"] = clean["abstract"].str.split().str.len()
    longest = clean.sort_values(["word_count", "id"], ascending=[False, True]).head(limit)
    if longest.empty:
        raise ValueError("Немає анотацій для поділу")
    return list(longest.itertuples(index=False))


def build_chunk_records(papers, max_words, overlap):
    records = {name: [] for name in INDEX_NAMES}
    splits = 0
    for number, paper in enumerate(papers):
        splits += sum(len(sentence.split()) > max_words for sentence in sentences(paper.abstract))
        for name, chunks in zip(INDEX_NAMES, (
            fixed_chunks(paper.abstract, max_words, overlap),
            sentence_chunks(paper.abstract, max_words),
        )):
            for chunk_index, chunk in enumerate(chunks):
                metadata = {"arxiv_id": paper.id, "title": paper.title, "chunk_text": chunk,
                            "chunk_index": chunk_index, "category": paper.category}
                if pd.notna(paper.year):
                    metadata["year"] = int(paper.year)
                records[name].append({"id": f"chunk_{number}_{chunk_index}", "metadata": metadata})
    return records, splits


def artifact_namespace(records, revision):
    payload = json.dumps({"model_revision": revision, "chunks": records}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return "chunks-" + sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_local(path, records, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(record, ensure_ascii=False) for record in records]
    with NamedTemporaryFile(dir=path.parent, prefix=".chunks-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, vectors=vectors, records=np.asarray(rows))
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def search_local(records, vectors, query, top_k):
    scores = vectors @ query
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(records[i]["metadata"], float(scores[i])) for i in order]


def log_results(name, hits):
    print(f"  {name}:")
    for rank, (metadata, score) in enumerate(hits, start=1):
        preview = metadata["chunk_text"]
        preview = preview[:180] + ("…" if len(preview) > 180 else "")
        print(f"    {rank}. {metadata['title']} | {metadata['arxiv_id']} | "
              f"{metadata.get('year', '—')} | {metadata['category']} | "
              f"чанк {int(metadata['chunk_index']) + 1} | cosine {score:.4f}")
        print(f"       {preview}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/arxiv_subset.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "embeddings/chunks")
    parser.add_argument("--backend", choices=("local", "pinecone"), default="local")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", action="append", help="Замінює типові запити; можна повторити.")
    args = parser.parse_args()
    if min(args.batch_size, args.limit, args.max_words, args.top_k) < 1 or not 0 <= args.overlap < args.max_words:
        parser.error("Розміри мають бути додатними й потрібно 0 ≤ overlap < max-words.")
    queries = tuple(q.strip() for q in (args.query or QUERIES))
    if not all(queries):
        parser.error("Запит не може бути порожнім.")

    from embedding_model import MODEL_REVISION, Specter2Encoder, format_paper

    papers = load_longest(args.data, args.limit)
    records, splits = build_chunk_records(papers, args.max_words, args.overlap)
    namespace = artifact_namespace(records, MODEL_REVISION)
    print(f"Режим: {args.backend}; відібрано {len(papers)} анотацій; "
          f"ліміт {args.max_words} слів, перекриття для fixed {args.overlap}.")
    print(f"Простір даних: {namespace}; довгих речень, які довелося розділити: {splits}.")
    encoder = Specter2Encoder(device=args.device)
    searchers = {}
    for name, group in records.items():
        text_batch = [format_paper(r["metadata"]["title"], r["metadata"]["chunk_text"]) for r in group]
        lengths = encoder.tokenizer(text_batch, truncation=False, return_length=True, verbose=False)["length"]
        over_budget = sum(n > encoder.max_seq_length for n in lengths)
        if over_budget:
            print(f"  {name}: модель скоротить {over_budget} входів понад {encoder.max_seq_length} токенів;")
            print("  зменш --max-words, якщо потрібно векторизувати фрагменти повністю.")
        vectors = encoder.encode(text_batch, batch_size=args.batch_size)
        counts = Counter(r["metadata"]["arxiv_id"] for r in group)
        print(f"{name}: {len(group)} чанків; {min(counts.values())}–{max(counts.values())} на статтю; "
              f"ембеддинги {vectors.shape}.")
        if args.backend == "local":
            path = args.output_dir / namespace / f"{name}.npz"
            save_local(path, group, vectors)
            print(f"  Збережено: {path}")
            searchers[name] = (group, vectors)
        else:
            from vector_store import pinecone_index, upsert_records, wait_for_count

            index = pinecone_index(name, create=True)
            payload = [{**record, "values": vector.tolist()} for record, vector in zip(group, vectors)]
            written = upsert_records(index, payload, namespace=namespace)
            if written != len(group):
                raise RuntimeError(f"{name}: Pinecone підтвердив {written} з {len(group)} записів")
            indexed = wait_for_count(index, len(group), namespace=namespace)
            print(f"  Pinecone: {indexed} векторів у просторі {namespace}.")
            searchers[name] = (index, {record["id"]: record["metadata"] for record in group})

    query_vectors = encoder.encode(queries, batch_size=args.batch_size, show_progress=False)
    for query, vector in zip(queries, query_vectors):
        print(f"\nЗапит: {query}")
        paper_sets = []
        for name, searcher in searchers.items():
            if args.backend == "local":
                hits = search_local(*searcher, vector, args.top_k)
            else:
                index, local_metadata = searcher
                response = index.query(vector=vector.tolist(), top_k=args.top_k, namespace=namespace,
                                       include_metadata=True, _request_timeout=30)
                hits = []
                for item in response.matches:
                    expected = local_metadata.get(item.id)
                    actual = item.metadata or {}
                    if not expected or any(actual.get(field) != expected[field]
                                           for field in ("arxiv_id", "chunk_index")):
                        raise ValueError(f"Результат Pinecone {item.id} не відповідає локальному набору чанків")
                    hits.append((expected, item.score))
            log_results(name, hits)
            paper_sets.append({metadata["arxiv_id"] for metadata, _ in hits})
        print(f"  Спільні статті в top-{args.top_k} двох стратегій: "
              f"{len(paper_sets[0] & paper_sets[1])}; різні: {len(paper_sets[0] ^ paper_sets[1])}.")


if __name__ == "__main__":
    main()

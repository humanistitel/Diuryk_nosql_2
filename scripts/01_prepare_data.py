"""Локальна адаптація стартового прикладу LMS для підготовки arXiv."""
import argparse
import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def extract_year(paper):
    versions = paper.get("versions") or []
    if versions:
        try:
            return parsedate_to_datetime(versions[0]["created"]).year
        except (TypeError, ValueError, KeyError):
            pass
    try:
        return datetime.strptime(paper.get("update_date") or "", "%Y-%m-%d").year
    except ValueError:
        return None


def clean(value):
    return " ".join((value or "").split())


def format_authors(paper):
    parsed = paper.get("authors_parsed") or []
    if not parsed:
        return clean(paper.get("authors"))
    return ", ".join(" ".join(piece for piece in (clean(e[0]), clean(e[1]) if len(e) > 1 else "") if piece)
                     for e in parsed[:10] if e and clean(e[0]))


def collect(input_file, limit):
    records = []
    with input_file.open(encoding="utf-8") as source:
        for line_no, line in enumerate(tqdm(source, desc="Читаємо JSONL"), start=1):
            if not line.strip():
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некоректний JSON у рядку {line_no}") from exc
            identifier = clean(paper.get("id"))
            title, abstract = clean(paper.get("title")), clean(paper.get("abstract"))
            if not (identifier and title and abstract):
                continue
            categories = clean(paper.get("categories")).split()
            records.append({
                "id": identifier, "title": title, "abstract": abstract,
                "authors": format_authors(paper), "year": extract_year(paper),
                "category": categories[0] if categories else "unknown",
            })
            if len(records) >= limit:
                break
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("arxiv-metadata-oai-snapshot.json"))
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_subset.parquet"))
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit має бути більше нуля")
    if args.output.exists():
        parser.error(f"Вихідний файл уже існує: {args.output}. Обери інший шлях.")
    records = collect(args.input, args.limit)
    if not records:
        parser.error("У файлі немає придатних записів")
    frame = pd.DataFrame(records)
    if frame["id"].duplicated().any():
        parser.error("У підвибірці є повторні arXiv ID; перевір джерело")
    frame["year"] = frame["year"].astype("Int64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Записано статей: {len(frame)}; без визначеного року: {frame['year'].isna().sum()}")
    print(frame["category"].value_counts().head(10).to_string())
    print(frame["year"].value_counts().sort_index().tail(10).to_string())
    print(f"Parquet: {args.output}")


if __name__ == "__main__":
    main()

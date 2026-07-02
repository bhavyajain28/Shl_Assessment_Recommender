"""CLI: build data/catalog.json from the live SHL site (or the historical
fallback if the live catalog table is unavailable -- see app/scraper.py).

Usage:
    python scripts/scrape_catalog.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.retriever import assessment_to_text
from app.scraper import build_catalog
from app.utils import setup_logging, slugify


def _write_documents(assessments, documents_dir: Path) -> None:
    documents_dir.mkdir(parents=True, exist_ok=True)
    for old in documents_dir.glob("*.txt"):
        old.unlink()
    for a in assessments:
        filename = f"{a.id}-{slugify(a.name)}.txt"
        with open(documents_dir / filename, "w", encoding="utf-8") as f:
            f.write(assessment_to_text(a) + f"\nurl: {a.url}\n")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    seed_csv = Path(__file__).resolve().parent.parent / "data" / "catalog_seed_historical.csv"
    assessments = build_catalog(seed_csv)

    settings.catalog_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.catalog_json_path, "w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in assessments], f, indent=2, ensure_ascii=False)

    documents_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    _write_documents(assessments, documents_dir)

    print(f"Wrote {len(assessments)} assessments to {settings.catalog_json_path}")
    print(f"Wrote {len(assessments)} per-assessment documents to {documents_dir}")
    by_type = {}
    for a in assessments:
        by_type[a.test_type or "(none)"] = by_type.get(a.test_type or "(none)", 0) + 1
    print("By test_type:", by_type)


if __name__ == "__main__":
    main()

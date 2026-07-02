"""CLI: precompute embeddings for data/catalog.json and persist a FAISS
index under data/vectorstore/, so the API doesn't pay embedding cost at
every cold start.

Usage:
    python scripts/build_vectorstore.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss

from app.config import get_settings
from app.embeddings import embed_texts
from app.models import Assessment
from app.retriever import assessment_to_text
from app.utils import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    with open(settings.catalog_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    assessments = [Assessment(**r) for r in raw]
    texts = [assessment_to_text(a) for a in assessments]

    print(f"Embedding {len(texts)} assessments with {settings.embedding_model} ...")
    vectors = embed_texts(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.vectorstore_dir / "faiss.index"))
    with open(settings.vectorstore_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": [a.id for a in assessments], "count": len(assessments)}, f)

    print(f"Wrote FAISS index ({vectors.shape[0]} vectors, dim={vectors.shape[1]}) to {settings.vectorstore_dir}")


if __name__ == "__main__":
    main()

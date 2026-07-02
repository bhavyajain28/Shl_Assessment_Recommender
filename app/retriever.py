"""Hybrid (semantic + lexical) retrieval over the SHL catalog.

Small catalog (~350 items), so we keep this simple and fast: a FAISS
flat-IP index for semantic similarity, fused via reciprocal-rank with a
BM25 lexical index for exact-name/acronym matches (crucial for queries like
"OPQ32r" that a 384-dim MiniLM embedding alone can under-rank against
semantically-similar-but-wrong items).
"""
from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.embeddings import embed_query, embed_texts
from app.models import Assessment

logger = logging.getLogger(__name__)

# Common SHL product acronyms whose letters don't literally appear in the
# catalog name -- used only to widen the query for get_by_name / comparisons.
ACRONYM_EXPANSIONS = {
    "opq": "occupational personality questionnaire opq",
    "gsa": "global skills assessment",
    "mq": "motivation questionnaire",
    "sjt": "situational judgement test",
    "ucf": "universal competency framework",
    "mfs": "360 multi-rater feedback system",
    "vadc": "virtual assessment and development center",
}


def assessment_to_text(a: Assessment) -> str:
    parts = [a.name, a.description, a.test_type_label]
    parts.extend(a.skills)
    parts.extend(a.job_levels)
    return " | ".join(p for p in parts if p)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().replace("/", " ").replace("-", " ").split() if t]


class Retriever:
    def __init__(self, catalog_path: Path | None = None, vectorstore_dir: Path | None = None):
        settings = get_settings()
        self.catalog_path = catalog_path or settings.catalog_json_path
        self.vectorstore_dir = vectorstore_dir or settings.vectorstore_dir
        self.assessments: list[Assessment] = self._load_catalog()
        self._texts = [assessment_to_text(a) for a in self.assessments]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts]) if self._texts else None
        self._index = self._load_or_build_index()

    def _load_catalog(self) -> list[Assessment]:
        if not self.catalog_path.exists():
            logger.warning("Catalog file not found at %s -- retriever will return no results", self.catalog_path)
            return []
        with open(self.catalog_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [Assessment(**r) for r in raw]

    def _load_or_build_index(self) -> faiss.Index | None:
        if not self.assessments:
            return None
        index_file = self.vectorstore_dir / "faiss.index"
        meta_file = self.vectorstore_dir / "meta.json"
        if index_file.exists() and meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("ids") == [a.id for a in self.assessments]:
                logger.info("Loaded FAISS index from %s (%d vectors)", index_file, meta.get("count", 0))
                return faiss.read_index(str(index_file))
            logger.warning("Vectorstore is stale relative to catalog.json -- rebuilding in memory")
        logger.info("Building FAISS index in memory for %d assessments", len(self.assessments))
        vectors = embed_texts(self._texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def get_all(self) -> list[Assessment]:
        return list(self.assessments)

    def search(self, query: str, k: int = 25) -> list[tuple[Assessment, float]]:
        if not self.assessments or self._index is None:
            return []
        k = min(k, len(self.assessments))

        query_vec = embed_query(query).reshape(1, -1)
        sem_scores, sem_idx = self._index.search(query_vec, k)
        semantic_rank = {int(idx): rank for rank, idx in enumerate(sem_idx[0]) if idx != -1}

        bm25_scores = self._bm25.get_scores(_tokenize(query)) if self._bm25 else np.zeros(len(self.assessments))
        lexical_order = np.argsort(-bm25_scores)[:k]
        lexical_rank = {int(idx): rank for rank, idx in enumerate(lexical_order)}

        candidates = set(semantic_rank) | set(lexical_rank)
        fused: list[tuple[int, float]] = []
        for idx in candidates:
            rr_semantic = 1.0 / (1 + semantic_rank.get(idx, k))
            rr_lexical = 1.0 / (1 + lexical_rank.get(idx, k))
            fused.append((idx, rr_semantic + rr_lexical))
        fused.sort(key=lambda pair: -pair[1])

        return [(self.assessments[idx], score) for idx, score in fused[:k]]

    def apply_filters(
        self,
        items: list[Assessment],
        test_types: list[str] | None = None,
        max_duration_minutes: int | None = None,
        remote_required: bool | None = None,
    ) -> list[Assessment]:
        filtered = items
        if test_types:
            wanted = {t.upper() for t in test_types}
            filtered = [a for a in filtered if not a.test_type or a.test_type in wanted]
        if max_duration_minutes is not None:
            filtered = [a for a in filtered if a.duration_minutes is None or a.duration_minutes <= max_duration_minutes]
        if remote_required:
            filtered = [a for a in filtered if a.remote_testing]
        return filtered

    def get_by_name(self, name: str, threshold: float = 0.72) -> Assessment | None:
        if not self.assessments or not name:
            return None
        query = name.strip().lower()
        expanded = ACRONYM_EXPANSIONS.get(query, query)

        for a in self.assessments:
            if a.name.strip().lower() == query:
                return a
        for a in self.assessments:
            if query in a.name.lower() or expanded in a.name.lower():
                return a

        best, best_score = None, 0.0
        for a in self.assessments:
            score = SequenceMatcher(None, expanded, a.name.lower()).ratio()
            if score > best_score:
                best, best_score = a, score
        return best if best_score >= threshold else None

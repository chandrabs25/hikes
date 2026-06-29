from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.data import DEFAULT_DECISION_META_DIR


DEFAULT_VECTOR_INDEX_DIR = DEFAULT_DECISION_META_DIR / "vector_index"
FULL_SECTION_CHUNK_TYPES = {"itinerary_full", "seasonality_full", "difficulty_full", "fitness_full"}
FULL_SECTION_MAX_PER_TREK = 1
COMPARISON_MAX_CHUNKS_PER_TREK = 2


class RetrievalIndexError(RuntimeError):
    """Raised when the local vector index is unavailable or invalid."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TrekVectorIndex:
    def __init__(self, index_dir: Path = DEFAULT_VECTOR_INDEX_DIR) -> None:
        self.index_dir = index_dir
        self.manifest: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self.embeddings is not None and bool(self.chunks)

    def _load(self) -> None:
        manifest_path = self.index_dir / "manifest.json"
        chunks_path = self.index_dir / "chunks.jsonl"
        embeddings_path = self.index_dir / "embeddings.npy"
        if not manifest_path.exists() or not chunks_path.exists() or not embeddings_path.exists():
            return
        self.manifest = json.loads(manifest_path.read_text())
        self.chunks = read_jsonl(chunks_path)
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        if self.embeddings.shape[0] != len(self.chunks):
            raise RetrievalIndexError(
                f"Vector index row mismatch: {self.embeddings.shape[0]} embeddings for {len(self.chunks)} chunks"
            )

    def reload(self) -> None:
        self.manifest = {}
        self.chunks = []
        self.embeddings = None
        self._load()

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        trek_ids: list[str],
        section_types: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if self.embeddings is None:
            raise RetrievalIndexError("Build data/decision_meta/vector_index before using retrieval chat.")
        allowed_treks = set(trek_ids)
        allowed_sections = set(section_types or [])
        row_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.get("trek_id") in allowed_treks
            and (not allowed_sections or chunk.get("section_type") in allowed_sections)
        ]
        if not row_indices:
            return []

        max_chunks_per_trek = COMPARISON_MAX_CHUNKS_PER_TREK if len(allowed_treks) > 1 else limit
        matrix = self.embeddings[row_indices]
        scores = matrix @ query_embedding
        ranked = np.argsort(scores)[::-1]
        results: list[dict[str, Any]] = []
        selected_counts_by_trek: dict[str, int] = {}
        full_section_counts_by_trek: dict[str, int] = {}
        for rank in ranked:
            row_index = row_indices[int(rank)]
            chunk = dict(self.chunks[row_index])
            trek_id = str(chunk.get("trek_id", ""))
            if selected_counts_by_trek.get(trek_id, 0) >= max_chunks_per_trek:
                continue
            is_full_section_chunk = str(chunk.get("section_type", "")) in FULL_SECTION_CHUNK_TYPES
            if is_full_section_chunk and full_section_counts_by_trek.get(trek_id, 0) >= FULL_SECTION_MAX_PER_TREK:
                continue
            chunk["score"] = float(scores[int(rank)])
            results.append(chunk)
            selected_counts_by_trek[trek_id] = selected_counts_by_trek.get(trek_id, 0) + 1
            if is_full_section_chunk:
                full_section_counts_by_trek[trek_id] = full_section_counts_by_trek.get(trek_id, 0) + 1
            if len(results) >= limit:
                break
        return results

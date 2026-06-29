#!/usr/bin/env python3
"""Build a joined trek knowledge catalog from slim and decision metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SLIM_META = Path("data/slim_meta")
DEFAULT_DECISION_META = Path("data/decision_meta")
DEFAULT_OUTPUT = Path("data/decision_meta/trek_knowledge_catalog.json")
SCHEMA_VERSION = "trek_knowledge_catalog_v1"

SECTION_USAGE = {
    "seasonality": ["llm_decision_meta_source", "embedding_source", "trek_discussion_source"],
    "difficulty": ["llm_decision_meta_source", "embedding_source", "trek_discussion_source"],
    "fitness": ["llm_decision_meta_source", "embedding_source", "trek_discussion_source"],
    "itinerary_day": ["embedding_source", "trek_discussion_source"],
    "faq": ["embedding_source", "trek_discussion_source"],
    "safety": ["embedding_source", "trek_discussion_source"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_slim_meta_by_trek(slim_dir: Path) -> dict[str, dict[str, Any]]:
    return {data["trek_id"]: data for data in (read_json(path) for path in sorted(slim_dir.glob("*.meta.json")))}


def load_profiles_by_trek(decision_dir: Path) -> dict[str, dict[str, Any]]:
    profile_dir = decision_dir / "profiles"
    if not profile_dir.exists():
        return {}
    return {data["trek_id"]: data for data in (read_json(path) for path in sorted(profile_dir.glob("*.json")))}


def load_chunks_by_trek(decision_dir: Path) -> dict[str, list[dict[str, Any]]]:
    chunks_by_trek: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in read_jsonl(decision_dir / "embedding_chunks.jsonl"):
        chunks_by_trek[chunk["trek_id"]].append(
            {
                "chunk_id": chunk["chunk_id"],
                "section_id": chunk["section_id"],
                "section_type": chunk["section_type"],
                "title": chunk["title"],
            }
        )
    return dict(chunks_by_trek)


def section_record(section: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    matching_chunks = [chunk["chunk_id"] for chunk in chunks if chunk["section_id"] == section["section_id"]]
    return {
        "section_id": section["section_id"],
        "section_type": section["section_type"],
        "source_field": section["source_field"],
        "title": section["title"],
        "rendered_support_status": section.get("rendered_support_status"),
        "usage": SECTION_USAGE.get(section["section_type"], ["source_text"]),
        "embedding_chunk_ids": matching_chunks,
        "text": section["text"],
    }


def sections_by_type(meta: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in meta.get("sections", []):
        grouped[section["section_type"]].append(section_record(section, chunks))
    return {key: grouped[key] for key in sorted(grouped)}


def build_catalog(slim_dir: Path, decision_dir: Path) -> dict[str, Any]:
    slim_by_trek = load_slim_meta_by_trek(slim_dir)
    profile_by_trek = load_profiles_by_trek(decision_dir)
    chunks_by_trek = load_chunks_by_trek(decision_dir)

    treks = []
    for trek_id in sorted(slim_by_trek):
        slim = slim_by_trek[trek_id]
        profile = profile_by_trek.get(trek_id)
        chunks = chunks_by_trek.get(trek_id, [])
        treks.append(
            {
                "trek_id": trek_id,
                "trek_title": slim.get("trek_title", ""),
                "source_url": slim.get("source_url", ""),
                "source_packet_file": slim.get("source_packet_file", ""),
                "quick_facts": {
                    "usage": ["deterministic_filter_source", "candidate_card_source"],
                    "data": slim.get("quick_facts", {}),
                },
                "decision_profile": {
                    "usage": ["runtime_candidate_profile"],
                    "data": profile,
                },
                "sections": sections_by_type(slim, chunks),
                "embedding_chunks": {
                    "usage": ["vector_embedding_inputs", "trek_discussion_retrieval"],
                    "count": len(chunks),
                    "chunks": chunks,
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_dirs": {
            "slim_meta": str(slim_dir),
            "decision_meta": str(decision_dir),
        },
        "trek_count": len(treks),
        "treks": treks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim-meta", type=Path, default=DEFAULT_SLIM_META)
    parser.add_argument("--decision-meta", type=Path, default=DEFAULT_DECISION_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = build_catalog(args.slim_meta, args.decision_meta)
    write_json(args.out, catalog)
    print(f"Wrote {args.out} with {catalog['trek_count']} trek(s)")


if __name__ == "__main__":
    main()

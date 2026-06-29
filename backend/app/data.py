from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECISION_META_DIR = REPO_ROOT / "data" / "decision_meta"


@dataclass(frozen=True)
class TrekRecord:
    trek_id: str
    trek_title: str
    source_url: str
    filter_record: dict[str, Any]
    profile: dict[str, Any] | None = None


def known_value(record: dict[str, Any], field: str) -> Any:
    value = record.get(field)
    if isinstance(value, dict) and value.get("status") == "known":
        return value.get("value")
    return None


def raw_fact(record: dict[str, Any], field: str) -> Any:
    return (record.get("raw_quick_facts") or {}).get(field)


class TrekRepository:
    def __init__(self, decision_meta_dir: Path = DEFAULT_DECISION_META_DIR) -> None:
        self.decision_meta_dir = decision_meta_dir
        self.records = self._load_records()
        self.profiles = self._load_profiles()
        self.treks = [
            TrekRecord(
                trek_id=record["trek_id"],
                trek_title=record["trek_title"],
                source_url=record.get("source_url", ""),
                filter_record=record,
                profile=self.profiles.get(record["trek_id"]),
            )
            for record in self.records
        ]

    def _load_records(self) -> list[dict[str, Any]]:
        path = self.decision_meta_dir / "filter_index.json"
        data = json.loads(path.read_text())
        return list(data["records"])

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        profiles_dir = self.decision_meta_dir / "profiles"
        profiles: dict[str, dict[str, Any]] = {}
        for path in profiles_dir.glob("*.json"):
            profile = json.loads(path.read_text())
            profiles[profile["trek_id"]] = profile
        return profiles

    def filter_options(self) -> dict[str, Any]:
        def values(field: str) -> list[Any]:
            return [
                value
                for record in self.records
                if (value := known_value(record, field)) is not None
            ]

        difficulty_order = ["Easy", "Easy-Moderate", "Moderate", "Moderate-Difficult", "Difficult"]
        present_difficulties = {raw_fact(record, "trek_difficulty") for record in self.records}
        pickup_cities = sorted(
            {
                pickup
                for record in self.records
                if (pickup := known_value(record, "pickup_city"))
            },
            key=str.casefold,
        )

        def range_for(field: str) -> dict[str, int | float | None]:
            known = values(field)
            return {"min": min(known) if known else None, "max": max(known) if known else None}

        return {
            "difficulty_buckets": [d for d in difficulty_order if d in present_difficulties],
            "pickup_cities": pickup_cities,
            "duration_days": range_for("duration_days"),
            "altitude_ft": range_for("highest_altitude_ft"),
            "age": {
                "min": min(values("min_age")) if values("min_age") else None,
                "max": max(values("max_age")) if values("max_age") else None,
            },
            "fitness_required_distance_km": range_for("fitness_required_distance_km"),
            "fitness_required_time_min": range_for("fitness_required_time_min"),
        }

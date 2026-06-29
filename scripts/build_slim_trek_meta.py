#!/usr/bin/env python3
"""Build slim metadata-ready trek source files for LLM extraction."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "slim_trek_meta_v1"
DEFAULT_INPUT = Path("data/source_packets")
DEFAULT_OUTPUT = Path("data/slim_meta")

FIELD_TO_SECTION_TYPE = {
    "best_time_section": "seasonality",
    "day_wise_itinerary": "itinerary_day",
    "difficulty_section": "difficulty",
    "faqs": "faq",
    "fitness_section": "fitness",
    "safety_section": "safety",
}

EXCLUDED_SOURCE_FIELDS = {"packing_section", "quick_itinerary", "quick_info"}
NOISE_LINE_PATTERN = re.compile(
    r"^(?:image|embed|video|iframe|transparent|previous slide|next slide|select month|generate my packing checklist)$",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+")
NUMERIC_LINE_PATTERN = re.compile(r"^[-+]?\d+(?:\.\d+)?$")
HEXISH_LINE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{12,}$")
DAYS_DISTANCE_PATTERN = re.compile(
    r"^(?P<days>\d+)\s+days?\s*(?:/\s*(?P<distance>\d+(?:\.\d+)?)\s*kms?)?$",
    re.IGNORECASE,
)
DISTANCE_PATTERN = re.compile(r"^(?P<distance>\d+(?:\.\d+)?)\s*kms?$", re.IGNORECASE)
ALTITUDE_PATTERN = re.compile(r"^(?P<altitude>\d[\d,]*)\s*(?:ft|feet)?$", re.IGNORECASE)

QUICK_FACT_ALIASES = {
    "trek_dificulty": "trek_difficulty",
    "trek_diffculty": "trek_difficulty",
    "pick_up_details": "pickup_details",
    "drop_off_details": "dropoff_details",
    "crosstrek_gear_rentals": "gear_rentals",
    "age_limit": "suitable_for",
}

QUICK_FACT_KEY_ORDER = [
    "trek_difficulty",
    "trek_duration",
    "total_trek_distance",
    "highest_altitude",
    "suitable_for",
    "basecamp",
    "region",
    "best_time",
    "accommodation_type",
    "fitness_criteria",
    "pickup_details",
    "dropoff_details",
    "pickup_point",
    "pickup_time",
    "dropoff_time",
    "cloakroom",
    "offloading",
    "packing_checklist",
    "gear_rentals",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    """Remove obvious embed/image noise without interpreting trek facts."""

    cleaned: list[str] = []
    previous = None
    for raw_line in (text or "").splitlines():
        line = URL_PATTERN.sub("", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if NOISE_LINE_PATTERN.fullmatch(line):
            continue
        if NUMERIC_LINE_PATTERN.fullmatch(line):
            continue
        if HEXISH_LINE_PATTERN.fullmatch(line) and not any(ch.isspace() for ch in line):
            continue
        if line == previous:
            continue
        cleaned.append(line)
        previous = line
    return "\n".join(cleaned).strip()


def normalize_distance(value: str) -> str:
    match = DISTANCE_PATTERN.fullmatch(value.strip())
    if not match:
        return value.strip()
    return f"{match.group('distance')} km"


def normalize_duration(value: str, total_distance: str | None = None) -> str:
    value = value.strip()
    match = DAYS_DISTANCE_PATTERN.fullmatch(value)
    if not match:
        return value

    days = match.group("days")
    distance = match.group("distance")
    if not distance and total_distance:
        distance_match = DISTANCE_PATTERN.fullmatch(total_distance.strip())
        if distance_match:
            distance = distance_match.group("distance")

    if distance:
        return f"{days} days / {distance} km"
    return f"{days} days"


def normalize_altitude(value: str) -> str:
    value = value.strip()
    match = ALTITUDE_PATTERN.fullmatch(value)
    if not match:
        return value
    altitude = int(match.group("altitude").replace(",", ""))
    return f"{altitude:,} ft"


def normalize_difficulty(value: str) -> str:
    return {
        "Easy Moderate": "Easy-Moderate",
        "Moderate - Difficult": "Moderate-Difficult",
    }.get(value.strip(), value.strip())


def combine_location_time(point: str | None, time: str | None) -> str | None:
    if point and time:
        return f"{point.strip()} at {time.strip()}"
    if point:
        return point.strip()
    return None


def normalize_quick_facts(raw_facts: dict[str, Any]) -> dict[str, Any]:
    """Normalize quick-info keys and display formats without inventing facts."""

    facts: dict[str, Any] = {}
    for key, value in raw_facts.items():
        facts[QUICK_FACT_ALIASES.get(key, key)] = value

    if "pick-up_and_drop-off" in facts:
        combined = facts.pop("pick-up_and_drop-off")
        facts.setdefault("pickup_details", combined)
        facts.setdefault("dropoff_details", combined)

    meeting_point = facts.pop("meeting_point", None)
    if meeting_point:
        facts.setdefault("pickup_details", meeting_point)

    pickup_details = facts.get("pickup_details")
    if not pickup_details:
        combined_pickup = combine_location_time(facts.get("pickup_point"), facts.get("pickup_time"))
        if combined_pickup:
            facts["pickup_details"] = combined_pickup
            facts.pop("pickup_point", None)
            facts.pop("pickup_time", None)

    dropoff_details = facts.get("dropoff_details")
    if not dropoff_details:
        drop_time = facts.get("dropoff_time") or facts.pop("drop_off_time", None)
        if drop_time:
            if "," in str(drop_time) or " at " in str(drop_time):
                facts["dropoff_details"] = str(drop_time).strip()
                facts.pop("dropoff_time", None)
            else:
                facts["dropoff_time"] = str(drop_time).strip()

    if "total_trek_distance" in facts:
        facts["total_trek_distance"] = normalize_distance(str(facts["total_trek_distance"]))
    if "trek_duration" in facts:
        facts["trek_duration"] = normalize_duration(str(facts["trek_duration"]), facts.get("total_trek_distance"))
    if "highest_altitude" in facts:
        facts["highest_altitude"] = normalize_altitude(str(facts["highest_altitude"]))
    if "trek_difficulty" in facts:
        facts["trek_difficulty"] = normalize_difficulty(str(facts["trek_difficulty"]))
    if "suitable_for" in facts:
        facts["suitable_for"] = str(facts["suitable_for"]).strip().replace(" - ", " to ")
    if "offloading" in facts:
        facts["offloading"] = str(facts["offloading"]).strip().replace("Not available", "Not Available")

    ordered: dict[str, Any] = {}
    for key in QUICK_FACT_KEY_ORDER:
        if key in facts:
            ordered[key] = facts[key]
    for key in sorted(facts):
        if key not in ordered:
            ordered[key] = facts[key]
    return ordered


def normalize_title(source_field: str, title: str) -> str:
    if source_field == "best_time_section":
        return "Seasonality"
    if source_field == "difficulty_section":
        return "Difficulty"
    if source_field == "fitness_section":
        return "Fitness"
    if source_field == "safety_section":
        return "Safety"
    return title


def rendered_support_status(section: dict[str, Any]) -> str:
    support = section.get("rendered_support") or {}
    return support.get("status") or "unknown"


def slim_section(section: dict[str, Any]) -> dict[str, Any] | None:
    source_field = section.get("source_field", "")
    if source_field in EXCLUDED_SOURCE_FIELDS:
        return None
    section_type = FIELD_TO_SECTION_TYPE.get(source_field)
    if not section_type:
        return None
    text = clean_text(section.get("text", ""))
    if not text:
        return None
    return {
        "section_id": section["packet_section_id"],
        "source_field": source_field,
        "section_type": section_type,
        "title": normalize_title(source_field, section.get("title", "")),
        "rendered_support_status": rendered_support_status(section),
        "text": text,
    }


def build_slim_meta(packet_path: Path) -> dict[str, Any]:
    packet = read_json(packet_path)
    quick_info_sections = [section for section in packet.get("source_sections", []) if section.get("source_field") == "quick_info"]
    quick_facts = {}
    if quick_info_sections:
        content = quick_info_sections[0].get("content")
        if isinstance(content, dict):
            quick_facts = normalize_quick_facts(content)

    sections = []
    for section in packet.get("source_sections", []):
        slim = slim_section(section)
        if slim:
            sections.append(slim)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "trek_id": packet["trek_id"],
        "trek_title": packet.get("trek_title", ""),
        "source_url": packet.get("source_url", ""),
        "source_packet_file": packet_path.name,
        "quick_facts": quick_facts,
        "sections": sections,
    }


def markdown(meta: dict[str, Any]) -> str:
    lines = [
        f"# {meta['trek_title']} - Slim Trek Meta",
        "",
        "> Metadata-ready source layer for LLM extraction. No facts are inferred here.",
        "",
        f"- Trek ID: `{meta['trek_id']}`",
        f"- Source URL: {meta['source_url']}",
        f"- Source packet: `{meta['source_packet_file']}`",
        f"- Sections: `{len(meta['sections'])}`",
        "",
        "## Quick Facts",
        "",
        "```json",
        json.dumps(meta["quick_facts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Sections",
        "",
    ]
    for section in meta["sections"]:
        lines.extend(
            [
                f"### {section['title']}",
                "",
                f"- Section ID: `{section['section_id']}`",
                f"- Source field: `{section['source_field']}`",
                f"- Section type: `{section['section_type']}`",
                f"- Rendered support: `{section['rendered_support_status']}`",
                "",
                "```text",
                section["text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def packet_paths(input_dir: Path, trek: str | None) -> list[Path]:
    if trek:
        path = input_dir / f"{trek}.packet.json"
        if not path.exists():
            raise SystemExit(f"Missing source packet: {path}")
        return [path]
    return sorted(input_dir.glob("*.packet.json"))


def build_outputs(input_dir: Path, out_dir: Path, trek: str | None = None) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for path in packet_paths(input_dir, trek):
        meta = build_slim_meta(path)
        metas.append(meta)
        stem = meta["trek_id"]
        write_json(out_dir / f"{stem}.meta.json", meta)
        (out_dir / f"{stem}.meta.md").write_text(markdown(meta), encoding="utf-8")

    jsonl_path = out_dir / "all_treks_meta.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for meta in metas:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "count": len(metas),
        "records": [
            {
                "trek_id": meta["trek_id"],
                "trek_title": meta["trek_title"],
                "source_url": meta["source_url"],
                "source_packet_file": meta["source_packet_file"],
                "section_count": len(meta["sections"]),
                "meta_json": f"{meta['trek_id']}.meta.json",
                "meta_md": f"{meta['trek_id']}.meta.md",
            }
            for meta in metas
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return metas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trek", help="Build one trek by id. Defaults to all source packets.")
    args = parser.parse_args()

    metas = build_outputs(args.input, args.out, args.trek)
    print(f"Wrote {len(metas)} slim trek meta file(s) to {args.out}")


if __name__ == "__main__":
    main()

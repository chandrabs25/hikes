#!/usr/bin/env python3
"""Extract lean decision metadata from slim trek meta files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_metadata_common import LlmProviderError, call_with_retries, env_model, require_api_key


SCHEMA_VERSION = "trek_decision_meta_v1"
PROFILE_SCHEMA_VERSION = "trek_decision_profile_v2"
PROMPT_VERSION = "trek_decision_meta_v1"
DEFAULT_INPUT = Path("data/slim_meta")
DEFAULT_OUTPUT = Path("data/decision_meta")
LLM_SECTION_TYPES = {"seasonality", "difficulty", "fitness"}
EMBED_SECTION_TYPES = {"seasonality", "difficulty", "fitness", "itinerary_day", "faq", "safety"}

DECISION_FIELDS = [
    "open_or_recommended_months",
    "avoid_months",
    "seasonal_experiences",
    "snow_window",
    "rain_or_monsoon_risk",
    "temperature_or_cold_notes",
    "view_or_landscape_notes",
    "seasonal_watchouts",
    "best_for",
    "not_ideal_for",
    "stated_difficulty",
    "experience_suitability",
    "terrain_challenges",
    "steep_or_strenuous_sections",
    "technical_or_equipment_notes",
    "altitude_risk_notes",
    "weather_risk_notes",
    "emergency_or_medical_notes",
    "group_watchouts",
    "fitness_benchmark",
    "approval_or_proof_requirements",
    "prep_window",
    "exceptions_or_age_notes",
    "risk_if_unfit",
]

SECTION_FIELD_MAP = {
    "seasonality": [
        "open_or_recommended_months",
        "avoid_months",
        "seasonal_experiences",
        "snow_window",
        "rain_or_monsoon_risk",
        "temperature_or_cold_notes",
        "view_or_landscape_notes",
        "seasonal_watchouts",
        "best_for",
        "not_ideal_for",
    ],
    "difficulty": [
        "stated_difficulty",
        "experience_suitability",
        "terrain_challenges",
        "steep_or_strenuous_sections",
        "technical_or_equipment_notes",
        "altitude_risk_notes",
        "weather_risk_notes",
        "emergency_or_medical_notes",
        "group_watchouts",
        "not_ideal_for",
    ],
    "fitness": [
        "fitness_benchmark",
        "approval_or_proof_requirements",
        "prep_window",
        "exceptions_or_age_notes",
        "risk_if_unfit",
        "group_watchouts",
    ],
}


class DecisionMetaError(RuntimeError):
    """Raised when decision metadata extraction cannot proceed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_slim_metas(input_dir: Path, trek: str | None = None) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.meta.json"))
    if trek:
        paths = [input_dir / f"{trek}.meta.json"]
    metas = []
    for path in paths:
        if not path.exists():
            raise DecisionMetaError(f"Missing slim meta file: {path}")
        metas.append(read_json(path))
    return metas


def known(value: Any) -> dict[str, Any]:
    return {"status": "known", "value": value}


def unknown(raw: Any = None) -> dict[str, Any]:
    result = {"status": "unknown", "value": None}
    if raw not in (None, ""):
        result["raw"] = raw
    return result


def parse_days_duration(value: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not value:
        return unknown(), unknown()
    days_match = re.search(r"(\d+)\s+days?", value, re.IGNORECASE)
    distance_match = re.search(r"/\s*(\d+(?:\.\d+)?)\s*km\b", value, re.IGNORECASE)
    days = known(int(days_match.group(1))) if days_match else unknown(value)
    distance = known(float(distance_match.group(1))) if distance_match else unknown(value)
    return days, distance


def parse_distance(value: str | None) -> dict[str, Any]:
    if not value:
        return unknown()
    match = re.search(r"(\d+(?:\.\d+)?)\s*km\b", value, re.IGNORECASE)
    return known(float(match.group(1))) if match else unknown(value)


def parse_altitude(value: str | None) -> dict[str, Any]:
    if not value:
        return unknown()
    match = re.search(r"(\d[\d,]*)\s*ft\b", value, re.IGNORECASE)
    return known(int(match.group(1).replace(",", ""))) if match else unknown(value)


def parse_age(value: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not value:
        return unknown(), unknown()
    nums = [int(item) for item in re.findall(r"\d+", value)]
    if not nums:
        return unknown(value), unknown(value)
    minimum = known(nums[0])
    maximum = known(nums[1]) if len(nums) > 1 and "to" in value.lower() else unknown()
    return minimum, maximum


def parse_fitness(value: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not value:
        return unknown(), unknown()
    match = re.search(r"(\d+)\s*km\s+in\s+(\d+)\s*mins?", value, re.IGNORECASE)
    if not match:
        return unknown(value), unknown(value)
    return known(int(match.group(1))), known(int(match.group(2)))


def parse_bool_available(value: str | None) -> dict[str, Any]:
    if not value:
        return unknown()
    normalized = value.strip().lower()
    if normalized == "available":
        return known(True)
    if normalized == "not available":
        return known(False)
    return unknown(value)


def parse_place_time(value: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not value:
        return unknown(), unknown()
    parts = re.split(r"\s+at\s+", value, maxsplit=1, flags=re.IGNORECASE)
    place = known(parts[0].strip()) if parts[0].strip() else unknown(value)
    time = known(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else unknown()
    return place, time


def build_filter_record(meta: dict[str, Any]) -> dict[str, Any]:
    facts = meta.get("quick_facts") or {}
    duration_days, distance_km = parse_days_duration(facts.get("trek_duration"))
    if distance_km["status"] == "unknown":
        distance_km = parse_distance(facts.get("total_trek_distance"))
    min_age, max_age = parse_age(facts.get("suitable_for"))
    fitness_distance, fitness_time = parse_fitness(facts.get("fitness_criteria"))
    pickup_city, pickup_time = parse_place_time(facts.get("pickup_details"))
    dropoff_city, dropoff_time = parse_place_time(facts.get("dropoff_details") or facts.get("dropoff_time"))
    return {
        "trek_id": meta["trek_id"],
        "trek_title": meta.get("trek_title", ""),
        "source_url": meta.get("source_url", ""),
        "raw_quick_facts": facts,
        "duration_days": duration_days,
        "distance_km": distance_km,
        "highest_altitude_ft": parse_altitude(facts.get("highest_altitude")),
        "min_age": min_age,
        "max_age": max_age,
        "fitness_required_distance_km": fitness_distance,
        "fitness_required_time_min": fitness_time,
        "pickup_city": pickup_city,
        "pickup_time": pickup_time,
        "dropoff_city": dropoff_city,
        "dropoff_time": dropoff_time,
        "offloading_available": parse_bool_available(facts.get("offloading")),
        "cloakroom_available": parse_bool_available(facts.get("cloakroom")),
        "accommodation_type": known(facts["accommodation_type"]) if facts.get("accommodation_type") else unknown(),
    }


def build_filter_index(metas: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source": "data/slim_meta",
        "count": len(metas),
        "records": [build_filter_record(meta) for meta in metas],
    }


def selected_sections(meta: dict[str, Any], types: set[str]) -> list[dict[str, Any]]:
    return [section for section in meta.get("sections", []) if section.get("section_type") in types]


def extraction_targets(metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for meta in metas:
        for section_type in ["seasonality", "difficulty", "fitness"]:
            matches = [section for section in meta.get("sections", []) if section.get("section_type") == section_type]
            if matches:
                section = dict(matches[0])
                section["trek_id"] = meta["trek_id"]
                section["trek_title"] = meta.get("trek_title", "")
                section["source_url"] = meta.get("source_url", "")
                targets.append(section)
    return targets


def chunk_id(trek_id: str, section_id: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", section_id).strip("-")
    return f"{trek_id}::{safe}::chunk-{index:03d}"


def split_by_headings(text: str) -> list[tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    heading_pattern = re.compile(r"^(?:[A-Z][A-Za-z0-9 '&()./-]{2,80}|The Terrain:|Terrain:|Weather:|Altitude:|Emergency Exits:|Fitness Target)$")
    chunks: list[tuple[str, list[str]]] = []
    current_title = "Overview"
    current_lines: list[str] = []
    for line in lines:
        if heading_pattern.match(line) and current_lines:
            chunks.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append((current_title, current_lines))
    if len(chunks) <= 1:
        return [("Overview", "\n".join(lines))]
    return [(title, "\n".join(chunk_lines)) for title, chunk_lines in chunks]


def embedding_chunks_for_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for section in selected_sections(meta, EMBED_SECTION_TYPES):
        section_type = section["section_type"]
        if section_type in {"faq", "itinerary_day", "safety"}:
            parts = [(section.get("title", section_type), section["text"])]
        else:
            parts = split_by_headings(section["text"])
        for index, (title, text) in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": chunk_id(meta["trek_id"], section["section_id"], index),
                    "trek_id": meta["trek_id"],
                    "trek_title": meta.get("trek_title", ""),
                    "section_id": section["section_id"],
                    "section_type": section_type,
                    "title": title if title != "Overview" else section.get("title", section_type),
                    "text": text,
                    "source_url": meta.get("source_url", ""),
                }
            )
    return chunks


def build_embedding_chunks(metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for meta in metas:
        chunks.extend(embedding_chunks_for_meta(meta))
    return chunks


def evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["section_id", "quote_or_summary", "confidence"],
        "properties": {
            "section_id": {"type": "string"},
            "quote_or_summary": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
    }


def decision_field_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["values", "evidence"],
        "properties": {
            "values": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": evidence_schema()},
        },
    }


def extraction_schema() -> dict[str, Any]:
    field_schema = decision_field_schema()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "prompt_version",
            "trek_id",
            "section_id",
            "section_type",
            "source_url",
            "fields",
            "needs_review",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "prompt_version": {"type": "string"},
            "trek_id": {"type": "string"},
            "section_id": {"type": "string"},
            "section_type": {"type": "string", "enum": sorted(LLM_SECTION_TYPES)},
            "source_url": {"type": "string"},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": DECISION_FIELDS,
                "properties": {field: field_schema for field in DECISION_FIELDS},
            },
            "needs_review": {"type": "array", "items": {"type": "string"}},
        },
    }


def empty_fields() -> dict[str, dict[str, Any]]:
    return {field: {"values": [], "evidence": []} for field in DECISION_FIELDS}


def section_messages(section: dict[str, Any]) -> list[dict[str, str]]:
    fields = SECTION_FIELD_MAP[section["section_type"]]
    return [
        {
            "role": "system",
            "content": (
                "You extract lean, evidence-backed trek decision metadata. Return JSON only. "
                "Extract only claims directly supported by the supplied section. Do not infer missing values. "
                "For fields without direct evidence, return empty values and empty evidence. "
                "Every non-empty values array must include evidence from the section. "
                "If you cannot provide evidence for a field, leave that field's values array empty."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "lean_decision_section_extraction",
                    "schema_version": SCHEMA_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "trek_id": section["trek_id"],
                    "trek_title": section["trek_title"],
                    "section_id": section["section_id"],
                    "section_type": section["section_type"],
                    "source_url": section["source_url"],
                    "fields_to_extract": fields,
                    "all_other_fields": "Return empty values/evidence.",
                    "section_text": section["text"],
                },
                ensure_ascii=False,
            ),
        },
    ]


def validate_extraction(data: dict[str, Any]) -> list[str]:
    errors = []
    for key in ["schema_version", "prompt_version", "trek_id", "section_id", "section_type", "source_url", "fields", "needs_review"]:
        if key not in data:
            errors.append(f"missing {key}")
    fields = data.get("fields")
    if not isinstance(fields, dict):
        return errors + ["fields must be object"]
    for field in DECISION_FIELDS:
        item = fields.get(field)
        if not isinstance(item, dict):
            errors.append(f"fields.{field}: missing object")
            continue
        values = item.get("values")
        evidence = item.get("evidence")
        if not isinstance(values, list):
            errors.append(f"fields.{field}.values: must be list")
        if not isinstance(evidence, list):
            errors.append(f"fields.{field}.evidence: must be list")
        if values and not evidence:
            errors.append(f"fields.{field}: missing evidence")
        for index, ref in enumerate(evidence or []):
            if not isinstance(ref, dict):
                errors.append(f"fields.{field}.evidence[{index}]: not object")
                continue
            for key in ["section_id", "quote_or_summary", "confidence"]:
                if not ref.get(key):
                    errors.append(f"fields.{field}.evidence[{index}]: missing {key}")
    return errors


def section_output_path(out_dir: Path, section: dict[str, Any]) -> Path:
    return out_dir / "section_extractions" / section["trek_id"] / f"{section['section_type']}.json"


def extract_section(section: dict[str, Any], args: argparse.Namespace, api_key: str, model: str) -> dict[str, Any]:
    path = section_output_path(args.out, section)
    if path.exists() and not args.force:
        print(f"  skip {section['trek_id']} {section['section_type']} (already complete)", flush=True)
        return read_json(path)
    messages = section_messages(section)
    total_attempts = 0
    parsed: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    errors: list[str] = []
    for validation_attempt in range(1, 4):
        parsed, raw, attempts = call_with_retries(
            api_key=api_key,
            model=model,
            messages=messages,
            schema_name="LeanDecisionSectionExtraction",
            schema=extraction_schema(),
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )
        total_attempts += attempts
        errors = validate_extraction(parsed)
        append_jsonl(
            args.out / "raw_responses.jsonl",
            {
                "stage": "section_extraction",
                "trek_id": section["trek_id"],
                "section_type": section["section_type"],
                "section_id": section["section_id"],
                "validation_attempt": validation_attempt,
                "valid": not errors,
                "validation_errors": errors,
                "completed_at": utc_now(),
                "response": raw,
            },
        )
        if not errors:
            break
        messages = messages + [
            {
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "repair_validation_errors",
                        "validation_errors": errors,
                        "instructions": (
                            "Return the same JSON schema corrected. For every field with non-empty values, "
                            "add evidence with section_id, quote_or_summary, and confidence. "
                            "If you cannot cite direct evidence from the section, set that field to "
                            "{\"values\": [], \"evidence\": []}."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    if errors or parsed is None:
        raise DecisionMetaError(f"Invalid extraction for {section['section_id']}: {errors}")
    write_json(path, parsed)
    print(f"  wrote {path} (api attempts={total_attempts})", flush=True)
    return parsed


def collect_values(extractions: list[dict[str, Any]], names: list[str]) -> list[str]:
    values = []
    for extraction in extractions:
        fields = extraction.get("fields", {})
        for name in names:
            for value in fields.get(name, {}).get("values", []):
                if value not in values:
                    values.append(value)
    return values


GENERIC_PROFILE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"safety is a shared responsibility",
        r"indiahikes team (?:is equipped with|carries).*medical kits",
        r"trekkers? must inform (?:your )?trek leader",
        r"please inform (?:your )?trek leader",
        r"track runs on strava",
        r"nike run club",
        r"take 3 screenshots",
        r"upload screenshots?",
        r"you do not enjoy the trek",
        r"miss out on .*transformative experience",
        r"cross-training workouts? will not be considered",
        r"only running, jogging or walking are accepted",
        r"technical team and trek leader monitor weather",
    ]
]


def normalize_for_dedupe(value: str) -> str:
    normalized = re.sub(r"\d+(?:\.\d+)?", "#", value.lower())
    normalized = re.sub(r"[^a-z#]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def is_generic_profile_value(value: str) -> bool:
    return any(pattern.search(value) for pattern in GENERIC_PROFILE_PATTERNS)


def compact_values(extractions: list[dict[str, Any]], names: list[str], limit: int = 5, drop_generic: bool = True) -> list[str]:
    values = []
    seen = set()
    for extraction in extractions:
        fields = extraction.get("fields", {})
        for name in names:
            for value in fields.get(name, {}).get("values", []):
                if drop_generic and is_generic_profile_value(value):
                    continue
                key = normalize_for_dedupe(value)
                if key and key not in seen:
                    values.append(value)
                    seen.add(key)
                if len(values) >= limit:
                    return values
    return values


def compact_evidence_refs(extractions: list[dict[str, Any]], field_names: list[str], limit: int = 12) -> list[dict[str, Any]]:
    evidence = []
    seen = set()
    for extraction in extractions:
        fields = extraction.get("fields", {})
        section_type = extraction.get("section_type", "")
        for name in field_names:
            for ref in fields.get(name, {}).get("evidence", []):
                key = (ref.get("section_id"), section_type, name)
                if key not in seen:
                    evidence.append(
                        {
                            "section_id": ref.get("section_id", ""),
                            "section_type": section_type,
                            "field_name": name,
                        }
                    )
                    seen.add(key)
                if len(evidence) >= limit:
                    return evidence
    return evidence


def consolidate_profile(meta: dict[str, Any], filter_record: dict[str, Any], extractions: list[dict[str, Any]]) -> dict[str, Any]:
    watchout_fields = [
        "seasonal_watchouts",
        "terrain_challenges",
        "steep_or_strenuous_sections",
        "technical_or_equipment_notes",
        "altitude_risk_notes",
        "weather_risk_notes",
        "group_watchouts",
        "risk_if_unfit",
        "not_ideal_for",
    ]
    evidence_fields = [
        "open_or_recommended_months",
        "avoid_months",
        "snow_window",
        "stated_difficulty",
        "experience_suitability",
        "terrain_challenges",
        "fitness_benchmark",
        "best_for",
        "not_ideal_for",
    ]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "trek_id": meta["trek_id"],
        "trek_title": meta.get("trek_title", ""),
        "source_url": meta.get("source_url", ""),
        "filter_index": compact_filter_record(filter_record),
        "candidate_profile": {
            "seasonality": {
                "recommended_months": compact_values(extractions, ["open_or_recommended_months"], limit=8, drop_generic=False),
                "avoid_months": compact_values(extractions, ["avoid_months"], limit=6, drop_generic=False),
                "snow_or_rain_notes": compact_values(extractions, ["snow_window", "rain_or_monsoon_risk"], limit=5),
                "temperature_notes": compact_values(extractions, ["temperature_or_cold_notes"], limit=5),
            },
            "difficulty": {
                "stated_difficulty": compact_values(extractions, ["stated_difficulty"], limit=3, drop_generic=False),
                "experience_suitability": compact_values(extractions, ["experience_suitability"], limit=5),
                "terrain_challenges": compact_values(extractions, ["terrain_challenges", "steep_or_strenuous_sections"], limit=5),
                "altitude_or_weather_risks": compact_values(extractions, ["altitude_risk_notes", "weather_risk_notes"], limit=5),
            },
            "fitness": {
                "fitness_benchmark": compact_values(extractions, ["fitness_benchmark"], limit=5, drop_generic=False),
                "approval_requirements": compact_values(extractions, ["approval_or_proof_requirements"], limit=5),
                "prep_window": compact_values(extractions, ["prep_window"], limit=5),
            },
            "experience_themes": compact_values(
                extractions,
                ["seasonal_experiences", "view_or_landscape_notes", "snow_window"],
                limit=6,
            ),
            "primary_watchouts": compact_values(extractions, watchout_fields, limit=6),
            "best_for": compact_values(extractions, ["best_for"], limit=5),
            "not_ideal_for": compact_values(extractions, ["not_ideal_for"], limit=5),
        },
        "evidence_refs": compact_evidence_refs(extractions, evidence_fields),
        "needs_review": sorted({item for extraction in extractions for item in extraction.get("needs_review", [])}),
    }


def compact_filter_value(item: dict[str, Any]) -> Any:
    if item.get("status") == "known":
        return item.get("value")
    return None


def compact_filter_record(filter_record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "duration_days",
        "distance_km",
        "highest_altitude_ft",
        "min_age",
        "max_age",
        "fitness_required_distance_km",
        "fitness_required_time_min",
        "pickup_city",
        "pickup_time",
        "dropoff_city",
        "dropoff_time",
        "offloading_available",
        "cloakroom_available",
        "accommodation_type",
    ]
    compact = {
        "trek_difficulty": filter_record.get("raw_quick_facts", {}).get("trek_difficulty"),
    }
    for key in keys:
        value = filter_record.get(key)
        compact[key] = compact_filter_value(value) if isinstance(value, dict) else None
    return compact


def write_profiles(out_dir: Path, metas: list[dict[str, Any]], filter_index: dict[str, Any]) -> list[dict[str, Any]]:
    filter_by_trek = {record["trek_id"]: record for record in filter_index["records"]}
    profiles = []
    for meta in metas:
        extraction_dir = out_dir / "section_extractions" / meta["trek_id"]
        extractions = []
        for section_type in ["seasonality", "difficulty", "fitness"]:
            path = extraction_dir / f"{section_type}.json"
            if path.exists():
                extractions.append(read_json(path))
        profile = consolidate_profile(meta, filter_by_trek[meta["trek_id"]], extractions)
        profiles.append(profile)
        write_json(out_dir / "profiles" / f"{meta['trek_id']}.json", profile)
    jsonl = out_dir / "all_trek_decision_profiles.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")
    return profiles


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(out_dir: Path, metas: list[dict[str, Any]], targets: list[dict[str, Any]], profiles: list[dict[str, Any]], model: str | None) -> None:
    write_json(
        out_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "model": model,
            "trek_count": len(metas),
            "llm_section_targets": {
                "seasonality": sum(1 for target in targets if target["section_type"] == "seasonality"),
                "difficulty": sum(1 for target in targets if target["section_type"] == "difficulty"),
                "fitness": sum(1 for target in targets if target["section_type"] == "fitness"),
            },
            "profile_count": len(profiles),
            "trek_ids": [meta["trek_id"] for meta in metas],
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trek")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def dry_run(metas: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    counts = {section_type: sum(1 for target in targets if target["section_type"] == section_type) for section_type in sorted(LLM_SECTION_TYPES)}
    non_targets = {"faq": 0, "itinerary_day": 0, "safety": 0, "quick_facts": 0}
    chunks = build_embedding_chunks(metas)
    print(f"Input treks: {len(metas)}")
    print(f"LLM extraction targets: {counts}")
    print(f"Non-LLM extraction targets: {non_targets}")
    print(f"Embedding chunks: {len(chunks)}")


def run(args: argparse.Namespace) -> None:
    if args.execute == args.dry_run:
        raise DecisionMetaError("Choose exactly one of --dry-run or --execute.")
    metas = load_slim_metas(args.input, args.trek)
    targets = extraction_targets(metas)
    if args.dry_run:
        dry_run(metas, targets)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    model = env_model()
    filter_index = build_filter_index(metas)
    write_json(args.out / "filter_index.json", filter_index)
    if not args.skip_embeddings:
        write_jsonl(args.out / "embedding_chunks.jsonl", build_embedding_chunks(metas))

    api_key = require_api_key()
    for index, section in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {section['trek_id']} {section['section_type']}", flush=True)
        extract_section(section, args, api_key, model)

    profiles = write_profiles(args.out, metas, filter_index)
    write_manifest(args.out, metas, targets, profiles, model)
    print(f"Wrote decision metadata under {args.out}")


def main() -> None:
    try:
        run(build_parser().parse_args())
    except LlmProviderError as exc:
        raise SystemExit(
            f"LLM provider request failed: {exc}\n"
            "If this is Fireworks HTTP 403, check that your API key has access to the selected model. "
            "You can override the model with FIREWORKS_MODEL=accounts/fireworks/models/<model-id>."
        ) from exc
    except DecisionMetaError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

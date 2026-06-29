#!/usr/bin/env python3
"""Shared helpers for evidence-backed LLM trek metadata extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_VERSION = "trek_metadata_v1"
DEFAULT_INPUT = Path("data/archive/prior_scrape_artifacts/full/trek_facts.json")
DEFAULT_OUTPUT = Path("data/archive/prior_scrape_artifacts/llm_metadata")
DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

NON_HIMALAYAN_UIDS = {
    "atsunta-pass-trek-omalo-shatili-georgia",
    "channarayana-durga-weekend-trek",
    "chhattisgarh-jungle-trek-guru-ghasi-das-national-park",
    "mount-rinjani-trek",
    "pench-tiger-trail",
}

CANONICAL_URL_BY_UID = {
    "deoriatal-chandrashila-trek": "https://indiahikes.com/deoriatal-chandrashila-trek",
    "pangarchulla-peak-trek": "https://indiahikes.com/pangarchulla-peak-trek",
}

CRITICAL_CATEGORIES = {
    "difficulty",
    "fitness",
    "age_and_group_suitability",
    "seasonality",
    "risks_and_watchouts",
}


class MetadataError(RuntimeError):
    """Raised when metadata extraction cannot proceed safely."""


class LlmProviderError(RuntimeError):
    """Raised with provider response details when an LLM request fails."""


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_section_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-")


def load_core_himalayan_treks(input_path: Path = DEFAULT_INPUT) -> list[dict[str, Any]]:
    records = read_json(input_path)
    by_uid: dict[str, dict[str, Any]] = {}
    for record in records:
        uid = record.get("uid")
        if not uid or uid in NON_HIMALAYAN_UIDS:
            continue
        canonical_url = CANONICAL_URL_BY_UID.get(uid)
        if canonical_url and record.get("url") != canonical_url:
            continue
        by_uid.setdefault(uid, record)
    return [by_uid[uid] for uid in sorted(by_uid)]


def select_treks(args: argparse.Namespace) -> list[dict[str, Any]]:
    treks = load_core_himalayan_treks(args.input)
    if args.trek:
        selected = [trek for trek in treks if trek.get("uid") == args.trek or trek.get("url", "").rstrip("/").endswith(args.trek)]
        if not selected:
            raise MetadataError(f"No core Himalayan trek found for {args.trek!r}")
        return selected
    if args.all_himalayan or args.dry_run:
        return treks
    raise MetadataError("Choose --trek TREK_ID, --all-himalayan, or --dry-run.")


def section_records_for_trek(trek: dict[str, Any], faq_group_size: int = 5) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    uid = trek["uid"]

    def add(section_type: str, source_field: str, title: str, text: str, index: int | None = None) -> None:
        if not text.strip():
            return
        suffix = f"{section_type}-{index:03d}" if index is not None else section_type
        sections.append(
            {
                "section_id": f"{uid}::{stable_section_id(suffix)}",
                "trek_id": uid,
                "trek_title": trek.get("title", ""),
                "source_url": trek.get("url", ""),
                "section_type": section_type,
                "source_field": source_field,
                "title": title,
                "text": text.strip(),
            }
        )

    add("quick_info", "quick_info", "Quick info", json.dumps(trek.get("quick_info") or {}, ensure_ascii=False, indent=2))
    add("difficulty", "difficulty_section", "Difficulty section", trek.get("difficulty_section") or "")
    add("best_time", "best_time_section", "Best time section", trek.get("best_time_section") or "")
    add("fitness", "fitness_section", "Fitness section", trek.get("fitness_section") or "")
    add("safety", "safety_section", "Safety section", trek.get("safety_section") or "")
    add("packing", "packing_section", "Packing section", trek.get("packing_section") or "")

    for idx, day in enumerate(trek.get("day_wise_itinerary") or [], start=1):
        add("itinerary_day", f"day_wise_itinerary[{idx - 1}]", f"Itinerary day {idx}", json.dumps(day, ensure_ascii=False, indent=2), idx)

    faqs = trek.get("faqs") or []
    for start in range(0, len(faqs), faq_group_size):
        group = faqs[start : start + faq_group_size]
        group_index = start // faq_group_size + 1
        add("faq_group", f"faqs[{start}:{start + len(group)}]", f"FAQ group {group_index}", json.dumps(group, ensure_ascii=False, indent=2), group_index)

    return sections


def evidence_ref_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_section", "source_field", "quote_or_summary", "confidence"],
        "properties": {
            "source_section": {"type": "string"},
            "source_field": {"type": "string"},
            "quote_or_summary": {"type": "string"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
    }


def section_schema() -> dict[str, Any]:
    ev = evidence_ref_schema()
    extracted_item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "field", "value", "status", "confidence", "evidence"],
        "properties": {
            "category": {"type": "string"},
            "field": {"type": "string"},
            "value": {},
            "status": {"type": "string", "enum": ["supported", "insufficient_evidence", "conflict"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "evidence": {"type": "array", "items": ev},
        },
    }
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
            "extracted_items",
            "needs_review",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "prompt_version": {"type": "string"},
            "trek_id": {"type": "string"},
            "section_id": {"type": "string"},
            "section_type": {"type": "string"},
            "source_url": {"type": "string"},
            "extracted_items": {"type": "array", "items": extracted_item},
            "needs_review": {"type": "array", "items": {"type": "string"}},
        },
    }


def profile_schema() -> dict[str, Any]:
    ev = evidence_ref_schema()
    metadata_field = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value", "status", "confidence", "evidence"],
        "properties": {
            "value": {},
            "status": {"type": "string", "enum": ["supported", "insufficient_evidence", "conflict"]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "evidence": {"type": "array", "items": ev},
        },
    }
    category = {"type": "object", "additionalProperties": metadata_field}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "prompt_version",
            "metadata_version",
            "trek_id",
            "source_url",
            "review_status",
            "identity",
            "geography_and_logistics",
            "difficulty",
            "fitness",
            "age_and_group_suitability",
            "seasonality",
            "experience_themes",
            "risks_and_watchouts",
            "itinerary_profile",
            "recommendation_profile",
            "comparison_axes",
            "needs_review",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "prompt_version": {"type": "string"},
            "metadata_version": {"type": "string"},
            "trek_id": {"type": "string"},
            "source_url": {"type": "string"},
            "review_status": {"type": "string", "enum": ["unreviewed", "reviewed", "rejected"]},
            "identity": category,
            "geography_and_logistics": category,
            "difficulty": category,
            "fitness": category,
            "age_and_group_suitability": category,
            "seasonality": category,
            "experience_themes": {"type": "array", "items": metadata_field},
            "risks_and_watchouts": {"type": "array", "items": metadata_field},
            "itinerary_profile": category,
            "recommendation_profile": category,
            "comparison_axes": category,
            "needs_review": {"type": "array", "items": {"type": "string"}},
        },
    }


def audit_schema() -> dict[str, Any]:
    issue = {
        "type": "object",
        "additionalProperties": False,
        "required": ["severity", "field_path", "issue", "recommendation"],
        "properties": {
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "field_path": {"type": "string"},
            "issue": {"type": "string"},
            "recommendation": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "prompt_version", "trek_id", "source_url", "issues", "summary", "passes_basic_audit"],
        "properties": {
            "schema_version": {"type": "string"},
            "prompt_version": {"type": "string"},
            "trek_id": {"type": "string"},
            "source_url": {"type": "string"},
            "issues": {"type": "array", "items": issue},
            "summary": {"type": "string"},
            "passes_basic_audit": {"type": "boolean"},
        },
    }


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": True,
        },
    }


def section_messages(section: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract evidence-backed trek metadata. Return JSON only. "
                "Extract only claims supported by the supplied section. Do not infer missing values. "
                "If evidence is insufficient, return status='insufficient_evidence'."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "section_extraction",
                    "schema_version": "v1",
                    "prompt_version": PROMPT_VERSION,
                    "trek_id": section["trek_id"],
                    "trek_title": section["trek_title"],
                    "section_id": section["section_id"],
                    "section_type": section["section_type"],
                    "source_field": section["source_field"],
                    "source_url": section["source_url"],
                    "section_text": section["text"],
                    "categories_to_extract": [
                        "identity",
                        "geography_and_logistics",
                        "difficulty",
                        "fitness",
                        "age_and_group_suitability",
                        "seasonality",
                        "experience_themes",
                        "risks_and_watchouts",
                        "itinerary_profile",
                        "recommendation_profile",
                        "comparison_axes",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def consolidation_messages(trek: dict[str, Any], section_outputs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You consolidate evidence-backed section extractions into one trek metadata profile. "
                "Return JSON only. Preserve evidence refs. Do not create values that are not present in section outputs. "
                "Default review_status must be 'unreviewed'. Mark conflicts and insufficient evidence explicitly."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "trek_consolidation",
                    "schema_version": "v1",
                    "prompt_version": PROMPT_VERSION,
                    "trek_id": trek["uid"],
                    "trek_title": trek.get("title", ""),
                    "source_url": trek.get("url", ""),
                    "section_extractions": section_outputs,
                    "required_top_level_categories": [
                        "identity",
                        "geography_and_logistics",
                        "difficulty",
                        "fitness",
                        "age_and_group_suitability",
                        "seasonality",
                        "experience_themes",
                        "risks_and_watchouts",
                        "itinerary_profile",
                        "recommendation_profile",
                        "comparison_axes",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]


def audit_messages(trek: dict[str, Any], profile: dict[str, Any], source_sections: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_sections = [
        {
            "section_id": item["section_id"],
            "section_type": item["section_type"],
            "source_field": item["source_field"],
            "text": item["text"][:6000],
        }
        for item in source_sections
    ]
    return [
        {
            "role": "system",
            "content": (
                "You audit trek metadata against source sections. Return JSON only. "
                "Find unsupported fields, contradictions, missing evidence, and safety/suitability weaknesses. "
                "Do not fix the metadata; report issues."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "metadata_audit",
                    "schema_version": "v1",
                    "prompt_version": PROMPT_VERSION,
                    "trek_id": trek["uid"],
                    "source_url": trek.get("url", ""),
                    "metadata_profile": profile,
                    "source_sections": compact_sections,
                },
                ensure_ascii=False,
            ),
        },
    ]


def chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    temperature: float,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": response_format(schema_name, schema),
    }
    request = urllib.request.Request(
        FIREWORKS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "indiahikes-decision-meta/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LlmProviderError(f"Fireworks HTTP {exc.code} {exc.reason}: {body}") from exc
    content = raw["choices"][0]["message"]["content"]
    return json.loads(content), raw


def call_with_retries(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    temperature: float,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    attempt = 0
    while True:
        attempt += 1
        try:
            parsed, raw = chat_completion(
                api_key=api_key,
                model=model,
                messages=messages,
                schema_name=schema_name,
                schema=schema,
                temperature=temperature,
                timeout=timeout,
            )
            return parsed, raw, attempt
        except (LlmProviderError, urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            if isinstance(exc, LlmProviderError):
                raise
            transient = isinstance(exc, (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError))
            if isinstance(exc, urllib.error.HTTPError):
                transient = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if attempt > retries or not transient:
                raise
            time.sleep(min(2**attempt, 30))


def validate_evidence_refs(evidence: Any, field_path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(evidence, list) or not evidence:
        return [f"{field_path}: missing evidence"]
    for index, ref in enumerate(evidence):
        if not isinstance(ref, dict):
            errors.append(f"{field_path}.evidence[{index}]: not an object")
            continue
        for key in ["source_section", "source_field", "quote_or_summary", "confidence"]:
            if not ref.get(key):
                errors.append(f"{field_path}.evidence[{index}]: missing {key}")
    return errors


def validate_section_extraction(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["schema_version", "prompt_version", "trek_id", "section_id", "section_type", "source_url", "extracted_items", "needs_review"]:
        if key not in data:
            errors.append(f"missing {key}")
    if not isinstance(data.get("extracted_items"), list):
        errors.append("extracted_items must be a list")
        return errors
    for index, item in enumerate(data["extracted_items"]):
        path = f"extracted_items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path}: not an object")
            continue
        for key in ["category", "field", "value", "status", "confidence", "evidence"]:
            if key not in item:
                errors.append(f"{path}: missing {key}")
        if item.get("status") == "supported":
            errors.extend(validate_evidence_refs(item.get("evidence"), path))
    return errors


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = [
        "schema_version",
        "prompt_version",
        "metadata_version",
        "trek_id",
        "source_url",
        "review_status",
        "identity",
        "geography_and_logistics",
        "difficulty",
        "fitness",
        "age_and_group_suitability",
        "seasonality",
        "experience_themes",
        "risks_and_watchouts",
        "itinerary_profile",
        "recommendation_profile",
        "comparison_axes",
        "needs_review",
    ]
    for key in required:
        if key not in profile:
            errors.append(f"missing {key}")
    if profile.get("review_status") != "unreviewed":
        errors.append("review_status must default to unreviewed")
    for category in CRITICAL_CATEGORIES:
        value = profile.get(category)
        if value in (None, {}, []):
            errors.append(f"{category}: missing category output")
    errors.extend(validate_metadata_container(profile.get("difficulty"), "difficulty", critical=True))
    errors.extend(validate_metadata_container(profile.get("fitness"), "fitness", critical=True))
    errors.extend(validate_metadata_container(profile.get("age_and_group_suitability"), "age_and_group_suitability", critical=True))
    errors.extend(validate_metadata_container(profile.get("seasonality"), "seasonality", critical=True))
    errors.extend(validate_metadata_container(profile.get("risks_and_watchouts"), "risks_and_watchouts", critical=True))
    return errors


def validate_metadata_container(value: Any, path: str, critical: bool) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        iterator = value.items()
    elif isinstance(value, list):
        iterator = [(str(index), item) for index, item in enumerate(value)]
    else:
        return [f"{path}: must be object or list"]
    for key, item in iterator:
        item_path = f"{path}.{key}"
        if not isinstance(item, dict):
            errors.append(f"{item_path}: not an object")
            continue
        status = item.get("status")
        if status == "supported":
            errors.extend(validate_evidence_refs(item.get("evidence"), item_path))
        elif critical and status not in {"insufficient_evidence", "conflict"}:
            errors.append(f"{item_path}: unsupported status {status!r}")
    return errors


def output_paths(base: Path, trek_id: str) -> dict[str, Path]:
    return {
        "sections_dir": base / "sections" / trek_id,
        "profile": base / "profiles" / f"{trek_id}.json",
        "audit": base / "audits" / f"{trek_id}.json",
        "raw": base / "raw_responses.jsonl",
        "section_jsonl": base / "section_extractions.jsonl",
        "manifest": base / "run_manifest.json",
    }


def env_model() -> str:
    return os.environ.get("FIREWORKS_MODEL", DEFAULT_MODEL)


def require_api_key() -> str:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise MetadataError("FIREWORKS_API_KEY is required for --execute.")
    return api_key

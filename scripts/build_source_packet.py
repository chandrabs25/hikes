#!/usr/bin/env python3
"""Build clean source packets from embedded trek facts plus rendered UI structure."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_metadata_common import load_core_himalayan_treks, read_json, write_json


DEFAULT_FACTS = Path("data/archive/prior_scrape_artifacts/full/trek_facts.json")
DEFAULT_STRUCTURE_DIR = Path("data/archive/prior_scrape_artifacts/rendered_structure")
DEFAULT_OUTPUT = Path("data/source_packets")
SCHEMA_VERSION = "source_packet_v1"

SOURCE_FIELD_BY_UI_KEY = {
    "quick_itinerary": ["quick_itinerary"],
    "detailed_itinerary": ["day_wise_itinerary"],
    "difficulty": ["difficulty_section"],
    "best_time": ["best_time_section"],
    "fitness": ["fitness_section"],
    "packing": ["packing_section"],
    "faqs": ["faqs"],
}

ALWAYS_INCLUDE_FIELDS = ["quick_info", "safety_section"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    lines = []
    previous = None
    for line in (value or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)


def select_trek(args: argparse.Namespace) -> dict[str, Any]:
    treks = load_core_himalayan_treks(args.facts)
    for trek in treks:
        if trek["uid"] == args.trek:
            return trek
    raise SystemExit(f"No core Himalayan trek found for {args.trek!r}")


def field_has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def source_field_to_sections(trek: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = trek.get(field)
    if not field_has_content(value):
        return []

    if field == "quick_info":
        return [
            {
                "source_field": field,
                "section_type": "structured_facts",
                "title": "Quick Info",
                "content": value,
                "text": json.dumps(value, ensure_ascii=False, indent=2),
            }
        ]

    if field in {"quick_itinerary", "day_wise_itinerary"}:
        sections = []
        for index, day in enumerate(value or [], start=1):
            title = day.get("place_title") or f"Day {day.get('day') or index}"
            sections.append(
                {
                    "source_field": field,
                    "section_type": "itinerary_day",
                    "title": title,
                    "content": day,
                    "text": json.dumps(day, ensure_ascii=False, indent=2),
                }
            )
        return sections

    if field == "faqs":
        sections = []
        for index, faq in enumerate(value or [], start=1):
            question = faq.get("question") or f"FAQ {index}"
            answer = faq.get("answer") or ""
            sections.append(
                {
                    "source_field": field,
                    "section_type": "faq",
                    "title": question,
                    "content": faq,
                    "text": normalize_text(f"Q: {question}\nA: {answer}"),
                }
            )
        return sections

    return [
        {
            "source_field": field,
            "section_type": "text_section",
            "title": field.replace("_", " ").title(),
            "content": value,
            "text": normalize_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)),
        }
    ]


def visible_support_for_field(field: str, structure: dict[str, Any]) -> dict[str, Any]:
    ui_keys = [key for key, fields in SOURCE_FIELD_BY_UI_KEY.items() if field in fields]
    matched_sections = [section for section in structure.get("ui_sections", []) if section.get("ui_key") in ui_keys]
    matched_anchors = [anchor for anchor in structure.get("anchors", []) if anchor.get("ui_key") in ui_keys]
    if field in ALWAYS_INCLUDE_FIELDS:
        return {
            "status": "included_source_field",
            "reason": "Included because it is source-derived context; not directly mapped from a rendered accordion.",
            "ui_sections": [],
            "anchors": [],
        }
    if matched_sections or matched_anchors:
        return {
            "status": "rendered_structure_matched",
            "ui_sections": matched_sections,
            "anchors": matched_anchors,
        }
    return {
        "status": "not_observed_in_rendered_structure",
        "ui_sections": [],
        "anchors": [],
    }


def build_packet(trek: dict[str, Any], structure: dict[str, Any]) -> dict[str, Any]:
    observed_ui_keys = {section.get("ui_key") for section in structure.get("ui_sections", [])}
    fields: list[str] = []
    for ui_key, source_fields in SOURCE_FIELD_BY_UI_KEY.items():
        if ui_key in observed_ui_keys or any(anchor.get("ui_key") == ui_key for anchor in structure.get("anchors", [])):
            fields.extend(source_fields)
    fields.extend(ALWAYS_INCLUDE_FIELDS)
    fields = sorted(dict.fromkeys(fields))

    source_sections = []
    for field in fields:
        for index, section in enumerate(source_field_to_sections(trek, field), start=1):
            source_sections.append(
                {
                    "packet_section_id": f"{trek['uid']}::{field}::{index:03d}",
                    **section,
                    "rendered_support": visible_support_for_field(field, structure),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "note": "Clean source packet. Facts come from source-derived trek_facts; Playwright contributes rendered structure only.",
        "generated_at": utc_now(),
        "trek_id": trek["uid"],
        "trek_title": trek.get("title", ""),
        "source_url": trek.get("url", ""),
        "has_complete_trek_information": bool(structure.get("has_complete_trek_information")),
        "rendered_structure_file": f"{structure.get('trek_id', trek['uid'])}.structure.json",
        "observed_ui_keys": sorted(key for key in observed_ui_keys if key),
        "source_fields_included": fields,
        "source_sections": source_sections,
    }


def packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        f"# {packet['trek_title']} - Source Packet Review",
        "",
        "> Facts below come from source-derived `trek_facts`; Playwright only confirms rendered UI structure.",
        "",
        f"- Trek ID: `{packet['trek_id']}`",
        f"- Source URL: {packet['source_url']}",
        f"- Has `#complete-trek-information`: `{packet['has_complete_trek_information']}`",
        f"- Observed UI keys: `{', '.join(packet['observed_ui_keys'])}`",
        f"- Source fields included: `{', '.join(packet['source_fields_included'])}`",
        f"- Source sections: `{len(packet['source_sections'])}`",
        "",
        "## Sections",
        "",
    ]
    for section in packet["source_sections"]:
        support = section["rendered_support"]
        lines.extend(
            [
                f"### {section['title']}",
                "",
                f"- Packet section ID: `{section['packet_section_id']}`",
                f"- Source field: `{section['source_field']}`",
                f"- Section type: `{section['section_type']}`",
                f"- Rendered support: `{support['status']}`",
            ]
        )
        if support.get("ui_sections"):
            labels = [item["label"] for item in support["ui_sections"][:5]]
            lines.append(f"- Matched UI labels: `{'; '.join(labels)}`")
        if support.get("anchors"):
            anchors = [item["id"] for item in support["anchors"][:5]]
            lines.append(f"- Matched anchors: `{'; '.join(anchors)}`")
        lines.extend(["", "```text", section["text"], "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trek", required=True)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    trek = select_trek(args)
    structure_path = args.structure_dir / f"{args.trek}.structure.json"
    if not structure_path.exists():
        raise SystemExit(f"Missing rendered structure file: {structure_path}")
    structure = read_json(structure_path)
    if not structure.get("has_complete_trek_information"):
        raise SystemExit(f"{args.trek} does not have #complete-trek-information in rendered structure")

    packet = build_packet(trek, structure)
    json_path = args.out / f"{args.trek}.packet.json"
    md_path = args.out / f"{args.trek}.packet.md"
    write_json(json_path, packet)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(packet_markdown(packet), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a human-review version of rendered trek source sections.

This is not metadata extraction. It only groups existing rendered source
sections into review buckets so the scrape quality can be inspected.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/archive/prior_scrape_artifacts/rendered_pages/pages")
DEFAULT_OUTPUT_DIR = Path("data/archive/prior_scrape_artifacts/reorganized")

BUCKETS = [
    "overview",
    "difficulty",
    "itinerary",
    "seasonality",
    "travel_logistics",
    "stay_camping_facilities",
    "fitness_age_suitability",
    "faq_general",
    "packing",
    "uncategorized",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    lines = []
    previous = None
    for line in text.splitlines():
        value = re.sub(r"\s+", " ", line).strip()
        if not value or value == previous:
            continue
        lines.append(value)
        previous = value
    return "\n".join(lines)


def is_teaser_section(section: dict[str, Any]) -> bool:
    heading = section.get("heading", "").lower()
    text = clean_text(section.get("text", ""))
    if section.get("char_count", 0) > 180:
        return False
    teaser_terms = [
        "what you can expect",
        "flight, train, bus",
        "stay, food, toilet",
        "budget and premium",
        "6-week plan",
        "get your questions answered",
    ]
    return any(term in heading or term in text.lower() for term in teaser_terms)


def classify_section(section: dict[str, Any]) -> str:
    heading = section.get("heading", "").lower()
    text = section.get("text", "").lower()
    combined = f"{heading}\n{text}"

    if "complete trek information" in heading:
        return "overview"
    if "packing" in heading or "packing checklist" in combined:
        return "packing"
    if re.search(r"\b(day\s*\d+|drive from|drive back|trek from|pickup day|pick-up day)\b", heading):
        return "itinerary"
    if any(term in heading for term in ["winter", "summer", "autumn", "spring", "best time", "snow", "temperature", "season"]):
        return "seasonality"
    if any(term in heading for term in ["how difficult", "difficulty", "easy", "moderate", "difficult", "terrain", "altitude", "weather"]):
        return "difficulty"
    if any(term in heading for term in ["travel", "reach", "return journey", "dehradun", "rishikesh", "atm", "luggage", "offloading"]):
        return "travel_logistics"
    if any(term in heading for term in ["camping", "stay options", "accommodation", "washroom", "toilet", "water sources", "mobile network", "charging"]):
        return "stay_camping_facilities"
    if any(term in heading for term in ["get fit", "fitness", "child", "age limit", "58 years", "high altitude trek"]):
        return "fitness_age_suitability"
    if heading.endswith("?") or "frequently asked" in heading:
        return "faq_general"
    return "uncategorized"


def reorganize(data: dict[str, Any]) -> dict[str, Any]:
    buckets = {bucket: [] for bucket in BUCKETS}
    skipped_teasers = []

    for section in data.get("sections", []):
        item = {
            "section_id": section["section_id"],
            "source_order": section["order"],
            "heading": section["heading"],
            "char_count": section["char_count"],
            "text_hash": section["text_hash"],
            "text": clean_text(section["text"]),
        }
        if is_teaser_section(section):
            skipped_teasers.append(item)
            continue
        buckets[classify_section(section)].append(item)

    return {
        "schema_version": "reorganized_rendered_trek_v1",
        "note": "Human-review grouping only. No facts are inferred or normalized here.",
        "trek_id": data["trek_id"],
        "trek_title": data.get("trek_title", ""),
        "source_url": data.get("source_url", ""),
        "source_page_file_scope": data.get("content_scope", ""),
        "source_section_count": data.get("section_count", 0),
        "grouped_section_count": sum(len(items) for items in buckets.values()),
        "skipped_teaser_count": len(skipped_teasers),
        "buckets": buckets,
        "skipped_teasers": skipped_teasers,
    }


def markdown_for_review(reorganized: dict[str, Any]) -> str:
    lines = [
        f"# {reorganized['trek_title']} - Reorganized Source Review",
        "",
        "> Human-review grouping only. No facts are inferred or normalized here.",
        "",
        f"- Trek ID: `{reorganized['trek_id']}`",
        f"- Source URL: {reorganized['source_url']}",
        f"- Source scope: `{reorganized['source_page_file_scope']}`",
        f"- Source sections: `{reorganized['source_section_count']}`",
        f"- Grouped sections: `{reorganized['grouped_section_count']}`",
        f"- Skipped teaser/menu sections: `{reorganized['skipped_teaser_count']}`",
        "",
        "## Bucket Summary",
        "",
    ]
    for bucket in BUCKETS:
        lines.append(f"- `{bucket}`: {len(reorganized['buckets'][bucket])}")
    lines.append("")

    for bucket in BUCKETS:
        sections = reorganized["buckets"][bucket]
        if not sections:
            continue
        lines.extend([f"## {bucket.replace('_', ' ').title()}", ""])
        for section in sections:
            lines.extend(
                [
                    f"### {section['heading']}",
                    "",
                    f"- Source order: `{section['source_order']}`",
                    f"- Section ID: `{section['section_id']}`",
                    f"- Characters: `{section['char_count']}`",
                    f"- Text hash: `{section['text_hash']}`",
                    "",
                    section["text"],
                    "",
                ]
            )

    if reorganized["skipped_teasers"]:
        lines.extend(["## Skipped Teaser/Menu Sections", ""])
        for section in reorganized["skipped_teasers"]:
            lines.extend(
                [
                    f"### {section['heading']}",
                    "",
                    f"- Source order: `{section['source_order']}`",
                    f"- Section ID: `{section['section_id']}`",
                    "",
                    section["text"],
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trek", required=True, help="Rendered trek id, e.g. dayara-bugyal-trek")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_path = args.input_dir / f"{args.trek}.json"
    if not input_path.exists():
        raise SystemExit(f"Missing rendered source file: {input_path}")

    reorganized = reorganize(read_json(input_path))
    json_path = args.out / f"{args.trek}.reorganized.json"
    md_path = args.out / f"{args.trek}.review.md"
    write_json(json_path, reorganized)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown_for_review(reorganized), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

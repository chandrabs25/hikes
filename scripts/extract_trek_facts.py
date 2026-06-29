#!/usr/bin/env python3
"""Extract structured trek facts from Indiahikes Next.js/Prismic page data."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any


NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)


def rich_text(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or "").strip())
            else:
                nested = rich_text(item)
                if nested:
                    parts.append(nested)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            return str(value.get("text") or "").strip()
        return "\n".join(rich_text(v) for v in value.values() if rich_text(v))
    if value is None:
        return ""
    return str(value).strip()


def load_next_data(raw_file: Path) -> dict[str, Any] | None:
    text = raw_file.read_text(encoding="utf-8", errors="ignore")
    match = NEXT_DATA_RE.search(text)
    if not match:
        return None
    return json.loads(html.unescape(match.group(1)))


def get_trek_data(next_data: dict[str, Any]) -> dict[str, Any] | None:
    page_props = next_data.get("props", {}).get("pageProps", {})
    trek_data = page_props.get("trekData")
    if isinstance(trek_data, dict) and trek_data.get("type") == "trek":
        return trek_data
    return None


def extract_quick_info(body: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for sl in body:
        if sl.get("slice_type") != "quick_info_section":
            continue
        for item in sl.get("items", []):
            title = str(item.get("title") or "").strip().lower().replace(" ", "_")
            content = rich_text(item.get("content"))
            if title and content:
                out[title] = content
    return out


def extract_slice_text(body: list[dict[str, Any]], slice_type: str) -> str:
    chunks = []
    for sl in body:
        if sl.get("slice_type") == slice_type:
            chunks.append(rich_text(sl.get("primary", {})))
            chunks.append(rich_text(sl.get("items", [])))
    return "\n".join(chunk for chunk in chunks if chunk)


def extract_quick_itinerary(body: list[dict[str, Any]]) -> list[dict[str, str]]:
    days = []
    for sl in body:
        if sl.get("slice_type") != "quick_itinerary":
            continue
        for item in sl.get("items", []):
            days.append(
                {
                    "day": rich_text(item.get("day_number_text")),
                    "heading": rich_text(item.get("heading1")),
                    "sub_heading": rich_text(item.get("sub_heading2")),
                    "details": rich_text(item.get("heading2")),
                }
            )
    return days


def extract_daywise(body: list[dict[str, Any]]) -> list[dict[str, str]]:
    days = []
    for sl in body:
        if sl.get("slice_type") != "day_wise_itinerary":
            continue
        day_num = sl.get("primary", {}).get("day_num")
        item = (sl.get("items") or [{}])[0]
        days.append(
            {
                "day": str(day_num or ""),
                "place_title": rich_text(item.get("place_title")),
                "difficulty": rich_text(item.get("difficulty")),
                "duration": rich_text(item.get("duration")),
                "altitude": rich_text(item.get("altitude")),
                "water_sources": rich_text(item.get("water_sources")),
                "description": rich_text(item.get("place_description_editor")),
            }
        )
    return days


def extract_faqs(body: list[dict[str, Any]]) -> list[dict[str, str]]:
    faqs = []
    for sl in body:
        if sl.get("slice_type") != "faq_about_trek":
            continue
        for item in sl.get("items", []):
            question = rich_text(item.get("question_heading"))
            answer = rich_text(item.get("answer_content"))
            if question:
                faqs.append({"question": question, "answer": answer})
    return faqs


def extract_trek(raw_file: Path, url: str) -> dict[str, Any] | None:
    next_data = load_next_data(raw_file)
    if not next_data:
        return None
    trek = get_trek_data(next_data)
    if not trek:
        return None

    data = trek.get("data", {})
    body = data.get("body", [])
    quick_info = extract_quick_info(body)
    similar = next_data.get("props", {}).get("pageProps", {}).get("similarTreks", [])
    similar_titles = []
    for item in similar:
        title = rich_text(item.get("data", {}).get("trek_title"))
        uid = item.get("uid", "")
        if title or uid:
            similar_titles.append({"title": title, "uid": uid})

    return {
        "uid": trek.get("uid", ""),
        "url": url,
        "title": rich_text(data.get("trek_title")),
        "active": data.get("active"),
        "first_publication_date": trek.get("first_publication_date", ""),
        "last_publication_date": trek.get("last_publication_date", ""),
        "meta_title": rich_text(data.get("meta_title")),
        "meta_description": rich_text(data.get("meta_description")),
        "quick_info": quick_info,
        "difficulty_section": extract_slice_text(body, "how_difficult_is_trek"),
        "best_time_section": extract_slice_text(body, "best_time_to_do_trek"),
        "fitness_section": extract_slice_text(body, "know_your_trek_fitness"),
        "packing_section": extract_slice_text(body, "trek_what_to_pack"),
        "safety_section": extract_slice_text(body, "safety_standards"),
        "quick_itinerary": extract_quick_itinerary(body),
        "day_wise_itinerary": extract_daywise(body),
        "faqs": extract_faqs(body),
        "similar_treks": similar_titles,
        "slice_types": [sl.get("slice_type") for sl in body],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/archive/prior_scrape_artifacts/manifest.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("data/archive/prior_scrape_artifacts/trek_facts.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/archive/prior_scrape_artifacts/trek_facts.csv"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    facts = []
    for row in rows:
        raw_file = row.get("raw_file")
        if not raw_file:
            continue
        fact = extract_trek(Path(raw_file), row["url"])
        if fact:
            facts.append(fact)

    args.out_json.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "uid",
            "title",
            "url",
            "difficulty",
            "duration",
            "highest_altitude",
            "suitable_for",
            "basecamp",
            "faq_count",
            "quick_itinerary_days",
            "day_wise_days",
            "similar_trek_count",
            "last_publication_date",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fact in facts:
            qi = fact["quick_info"]
            writer.writerow(
                {
                    "uid": fact["uid"],
                    "title": fact["title"],
                    "url": fact["url"],
                    "difficulty": qi.get("trek_difficulty", ""),
                    "duration": qi.get("trek_duration", ""),
                    "highest_altitude": qi.get("highest_altitude", ""),
                    "suitable_for": qi.get("suitable_for", ""),
                    "basecamp": qi.get("basecamp", ""),
                    "faq_count": len(fact["faqs"]),
                    "quick_itinerary_days": len(fact["quick_itinerary"]),
                    "day_wise_days": len(fact["day_wise_itinerary"]),
                    "similar_trek_count": len(fact["similar_treks"]),
                    "last_publication_date": fact["last_publication_date"],
                }
            )

    print(f"Extracted {len(facts)} trek pages")
    print(f"Wrote {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()

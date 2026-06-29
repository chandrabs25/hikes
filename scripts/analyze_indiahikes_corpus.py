#!/usr/bin/env python3
"""Analyze the collected Indiahikes corpus before knowledge graph design."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FIELD_PATTERNS = {
    "trek_difficulty": re.compile(r"\bTREK DIFFICULTY\b|Easy[- ]?Moderate|Moderate[- ]?Difficult|Difficult", re.I),
    "trek_duration": re.compile(r"\bTREK DURATION\b|\b\d+\s*days?\s*/\s*[\d.]+\s*kms?\b", re.I),
    "highest_altitude": re.compile(r"\bHIGHEST ALTITUDE\b|\b\d{1,2},?\d{3}\s*ft\b|\b\d{4,5}\s*feet\b", re.I),
    "suitable_for": re.compile(r"\bSUITABLE FOR\b|\bSuitable for\b", re.I),
    "basecamp": re.compile(r"\bBASECAMP\b|\bBasecamp\b", re.I),
    "itinerary": re.compile(r"\bDay\s+\d+\b|\bTrek Distance\b|\bTrek Duration\b|Altitude Gain|Altitude Loss", re.I),
    "fitness": re.compile(r"\bfitness\b|\bjog\b|\bcardio\b|\btraining\b|\bprepare\b", re.I),
    "gear": re.compile(r"\bgear\b|\btrekking shoes\b|\bbackpack\b|\brain cover\b|\bmicrospikes\b", re.I),
    "safety": re.compile(r"\bsafety\b|\bAMS\b|\boxygen\b|\bpulse\b|\brisk\b|\bemergency\b", re.I),
    "season": re.compile(r"\bseason\b|\bmonth\b|\bDecember\b|\bJanuary\b|\bJune\b|\bSeptember\b|\bwinter\b|\bsummer\b|monsoon", re.I),
    "faq": re.compile(r"\bFAQ\b|\bexpand accordion\b|\bfrequently asked\b", re.I),
}


def classify_url(url: str) -> str:
    if "/blog/" in url:
        return "blog"
    if any(part in url for part in ["/faq", "/green-trails", "/meet-the-team"]):
        return "site_reference"
    if url.rstrip("/") == "https://indiahikes.com":
        return "home"
    return "trek_or_landing"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/archive/prior_scrape_artifacts/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    status_counts = Counter(row["status"] or "ERR" for row in rows)
    type_counts = Counter(classify_url(row["url"]) for row in rows)
    challenge_count = sum(1 for row in rows if row["challenge"] == "yes")
    available = [row for row in rows if row.get("text_file")]

    field_hits: dict[str, dict[str, bool]] = {}
    field_counts = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    word_counts = {}

    for row in available:
        text = Path(row["text_file"]).read_text(encoding="utf-8", errors="ignore")
        word_counts[row["url"]] = len(re.findall(r"\w+", text))
        field_hits[row["url"]] = {}
        for name, pattern in FIELD_PATTERNS.items():
            hit = bool(pattern.search(text))
            field_hits[row["url"]][name] = hit
            if hit:
                field_counts[name] += 1
                if len(samples[name]) < 5:
                    samples[name].append(row["url"])

    summary = {
        "pages_in_manifest": len(rows),
        "pages_with_text": len(available),
        "challenge_pages": challenge_count,
        "status_counts": dict(status_counts),
        "content_type_counts": dict(type_counts),
        "field_signal_counts": dict(field_counts),
        "field_signal_samples": samples,
        "word_counts": word_counts,
    }

    (args.out / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Indiahikes Corpus Analysis",
        "",
        "## Fetch Summary",
        f"- Pages attempted: {len(rows)}",
        f"- Pages converted to text: {len(available)}",
        f"- Challenge/blocked pages: {challenge_count}",
        f"- HTTP statuses: {dict(status_counts)}",
        "",
        "## URL Types",
    ]
    for content_type, count in type_counts.most_common():
        lines.append(f"- {content_type}: {count}")

    lines.extend(["", "## Field Signals In Available Text"])
    if available:
        for field, count in field_counts.most_common():
            lines.append(f"- {field}: {count}/{len(available)} pages")
    else:
        lines.append("- No page text available yet. Fetches were blocked/challenged or no HTML text was collected.")

    lines.extend(
        [
            "",
            "## Recommended Next Collection Step",
            "Use an approved URL export, manually saved HTML pages, or a partner-approved crawl path from Indiahikes.",
            "Do not use Cloudflare challenge pages as source content.",
        ]
    )
    (args.out / "corpus_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out / 'corpus_analysis.md'} and {args.out / 'corpus_summary.json'}")


if __name__ == "__main__":
    main()

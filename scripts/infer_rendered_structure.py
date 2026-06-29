#!/usr/bin/env python3
"""Infer the rendered Indiahikes trek-information UI structure with Playwright.

This script does not scrape facts. It records which sections the public page
exposes under #complete-trek-information so source packets can be built from
clean embedded source data with a rendered-UI audit trail.
"""

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


DEFAULT_INPUT = Path("data/archive/prior_scrape_artifacts/full/trek_facts.json")
DEFAULT_OUTPUT = Path("data/archive/prior_scrape_artifacts/rendered_structure")
USER_AGENT = "TrekPathStructureAudit/0.1 (+local product research; rendered public pages)"
SCHEMA_VERSION = "rendered_structure_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def stable_key(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.lower()).strip("-")
    safe = re.sub(r"-{2,}", "-", safe)
    return safe or "section"


def import_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Playwright is not installed. Run:\n"
            "  python3 -m pip install playwright\n"
            "  python3 -m playwright install chromium"
        ) from exc
    return sync_playwright


def select_treks(args: argparse.Namespace) -> list[dict[str, Any]]:
    treks = load_core_himalayan_treks(args.input)
    if args.trek:
        selected = [trek for trek in treks if trek["uid"] == args.trek]
        if not selected:
            raise SystemExit(f"No core Himalayan trek found for {args.trek!r}")
        return selected
    if args.limit:
        return treks[: args.limit]
    if args.all_himalayan:
        return treks
    raise SystemExit("Choose --trek TREK_ID or --all-himalayan")


def classify_ui_section(heading: str, anchor_id: str = "") -> str:
    value = f"{heading} {anchor_id}".lower()
    if "quick itinerary" in value:
        return "quick_itinerary"
    if "each day" in value or "detailed-itinerary" in value:
        return "detailed_itinerary"
    if "difficult" in value or "trek-difficulty" in value:
        return "difficulty"
    if "best time" in value:
        return "best_time"
    if "travel" in value:
        return "travel"
    if "camping experience" in value:
        return "camping"
    if "stay options" in value:
        return "stay"
    if "packing" in value or "what-to-pack" in value:
        return "packing"
    if "frequently asked" in value or value.strip() == "faq" or " faq" in value:
        return "faqs"
    if "get fit" in value or "fitness" in value or "getting-fit" in value:
        return "fitness"
    return "other"


def structure_script() -> str:
    return r"""
() => {
  const clean = value => (value || '').replace(/\s+/g, ' ').trim();
  const root = document.querySelector('#complete-trek-information');
  if (!root) {
    return {
      has_complete_trek_information: false,
      title: document.title || '',
      url: location.href,
      h1: clean(document.querySelector('h1')?.innerText || ''),
      sections: [],
      anchors: []
    };
  }

  function visible(el) {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
  }

  const anchors = Array.from(root.querySelectorAll('[id]')).map(el => {
    const rect = el.getBoundingClientRect();
    return {
      id: el.id,
      tag: el.tagName.toLowerCase(),
      text: clean(el.innerText).slice(0, 220),
      top: Math.round(rect.top + window.scrollY),
      visible: visible(el)
    };
  }).filter(item => item.id);

  const controls = Array.from(root.querySelectorAll('button,a,h2,h3,h4')).map(el => {
    const rect = el.getBoundingClientRect();
    const text = clean(el.innerText || el.textContent || '');
    return {
      tag: el.tagName.toLowerCase(),
      text,
      href: el.href || el.getAttribute('href') || '',
      id: el.id || '',
      class_name: typeof el.className === 'string' ? el.className : '',
      top: Math.round(rect.top + window.scrollY),
      visible: visible(el)
    };
  }).filter(item => item.text);

  return {
    has_complete_trek_information: true,
    title: document.title || '',
    url: location.href,
    h1: clean(document.querySelector('h1')?.innerText || ''),
    root_text_preview: clean(root.innerText).slice(0, 1200),
    anchors,
    controls
  };
}
"""


def infer_structure(page: Any, trek: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    page.goto(trek["url"], wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
    except Exception:
        pass
    raw = page.evaluate(structure_script())
    if not raw.get("has_complete_trek_information"):
        return {
            "schema_version": SCHEMA_VERSION,
            "trek_id": trek["uid"],
            "trek_title": trek.get("title", ""),
            "source_url": trek.get("url", ""),
            "inferred_at": utc_now(),
            "has_complete_trek_information": False,
            "page_title": raw.get("title", ""),
            "h1": raw.get("h1", ""),
            "ui_sections": [],
            "anchors": [],
            "root_text_preview": "",
        }

    ui_sections: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw.get("controls") or []:
        if not item.get("visible"):
            continue
        text = normalize_text(item.get("text", ""))
        if not text:
            continue
        if item["tag"] in {"button", "h2", "h3", "h4", "a"}:
            key = (item["tag"], text)
            if key in seen:
                continue
            seen.add(key)
            ui_key = classify_ui_section(text, item.get("id", ""))
            if ui_key == "other" and item["tag"] == "a":
                continue
            ui_sections.append(
                {
                    "ui_section_id": f"{trek['uid']}::ui::{len(ui_sections) + 1:03d}-{stable_key(text)[:60]}",
                    "ui_key": ui_key,
                    "label": text,
                    "tag": item["tag"],
                    "href": item.get("href", ""),
                    "dom_id": item.get("id", ""),
                    "top": item.get("top", 0),
                }
            )

    anchors = []
    for anchor in raw.get("anchors") or []:
        anchors.append(
            {
                "id": anchor["id"],
                "tag": anchor["tag"],
                "ui_key": classify_ui_section(anchor.get("text", ""), anchor["id"]),
                "text_preview": anchor.get("text", ""),
                "top": anchor.get("top", 0),
                "visible": bool(anchor.get("visible")),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "trek_id": trek["uid"],
        "trek_title": trek.get("title", ""),
        "source_url": trek.get("url", ""),
        "inferred_at": utc_now(),
        "has_complete_trek_information": True,
        "page_title": raw.get("title", ""),
        "h1": raw.get("h1", ""),
        "root_text_preview": raw.get("root_text_preview", ""),
        "anchors": anchors,
        "ui_sections": ui_sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trek")
    parser.add_argument("--all-himalayan", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    treks = select_treks(args)
    sync_playwright = import_playwright()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for index, trek in enumerate(treks, start=1):
            context = browser.new_context(viewport={"width": 1440, "height": 1200}, user_agent=USER_AGENT, locale="en-IN")
            page = context.new_page()
            try:
                structure = infer_structure(page, trek, args.timeout * 1000)
                write_json(args.out / f"{trek['uid']}.structure.json", structure)
                status = "ok" if structure["has_complete_trek_information"] else "skipped_missing_anchor"
                print(f"[{index}/{len(treks)}] {status} {trek['uid']} ui_sections={len(structure['ui_sections'])}", flush=True)
                records.append({"trek_id": trek["uid"], "url": trek["url"], "status": status, "ui_sections": len(structure["ui_sections"])})
            except Exception as exc:
                print(f"[{index}/{len(treks)}] error {trek['uid']}: {exc!r}", flush=True)
                records.append({"trek_id": trek["uid"], "url": trek["url"], "status": "error", "error": repr(exc)})
            finally:
                context.close()
        browser.close()

    write_json(
        args.out / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "source": "playwright_rendered_structure",
            "count": len(records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scrape rendered, visible Indiahikes trek page text with Playwright.

This creates a source layer that reflects what a visitor can actually read in
the browser. Embedded Next.js/Prismic data remains useful as a backup, but it
can contain stale hidden text; this script treats the rendered DOM as primary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from llm_metadata_common import load_core_himalayan_treks, read_json, write_json


DEFAULT_INPUT = Path("data/archive/prior_scrape_artifacts/full/trek_facts.json")
DEFAULT_OUTPUT = Path("data/archive/prior_scrape_artifacts/rendered_pages")
SCHEMA_VERSION = "rendered_visible_dom_v1"
USER_AGENT = "TrekPathVisibleSource/0.1 (+local product research; rendered public pages)"
CONTENT_SCOPE = "complete_trek_information"
EXPANDABLE_BUTTON_PATTERN = re.compile(
    r"(toggle accordion|expand accordion|read more|expand overview|quick itinerary|complete day-wise guide)",
    re.IGNORECASE,
)
COMPLETE_TREK_INFORMATION_PATTERN = re.compile(r"\bcomplete\s+trek\s+information\b", re.IGNORECASE)
END_OF_COMPLETE_INFORMATION_PATTERN = re.compile(
    r"^(the indiahikes spirit of trekking|photo gallery|read more on|trek trivia|treks by region|contact us)\b",
    re.IGNORECASE,
)


class MissingCompleteTrekInformationAnchor(RuntimeError):
    """Raised when a page does not expose the required trek information anchor."""


class EmptyScopedExtraction(RuntimeError):
    """Raised when the required source container yields no usable sections."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.lower()).strip("-")
    safe = re.sub(r"-{2,}", "-", safe)
    return safe or "section"


def normalize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def content_zone_for_heading(heading: str) -> str:
    value = heading.lower()
    if any(term in value for term in ["treks by region", "contact us"]):
        return "global_page_chrome"
    if any(term in value for term in ["read more on", "trek trivia"]):
        return "related_content"
    if any(term in value for term in ["trekkers share", "loved the"]):
        return "testimonial"
    if any(
        term in value
        for term in [
            "complete trek information",
            "how difficult",
            "plan your travel",
            "camping experience",
            "stay options",
            "get fit",
            "quick itinerary",
            "each day",
            "detailed itinerary",
            "highlights",
        ]
    ):
        return "trek_information"
    return "trek_overview"


def load_active_treks(path: Path) -> list[dict[str, Any]]:
    records = read_json(path)
    by_uid: dict[str, dict[str, Any]] = {}
    for record in records:
        uid = record.get("uid")
        if uid and record.get("active") is True:
            by_uid.setdefault(uid, record)
    return [by_uid[uid] for uid in sorted(by_uid)]


def select_treks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_active:
        treks = load_active_treks(args.input)
    else:
        treks = load_core_himalayan_treks(args.input)

    if args.trek:
        wanted = set(args.trek)
        treks = [trek for trek in treks if trek.get("uid") in wanted]
        missing = wanted - {trek.get("uid") for trek in treks}
        if missing:
            raise SystemExit(f"No selected active trek found for: {', '.join(sorted(missing))}")

    if args.limit:
        treks = treks[: args.limit]
    return treks


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


def visible_extraction_script() -> str:
    return r"""
() => {
  const excludedSelector = [
    'script',
    'style',
    'noscript',
    'svg',
    'iframe',
    'header',
    'footer',
    'nav',
    '[role="banner"]',
    '[role="navigation"]',
    '[aria-hidden="true"]'
  ].join(',');

  function isVisible(el) {
    if (el.closest(excludedSelector)) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function clean(value) {
    return (value || '').replace(/\s+/g, ' ').trim();
  }

  const root = document.querySelector('#complete-trek-information');
  if (!root) {
    return {
      missing_complete_trek_information_anchor: true,
      title: document.title || '',
      url: location.href,
      canonical_url: document.querySelector('link[rel="canonical"]')?.href || '',
      meta_description: document.querySelector('meta[name="description"]')?.content || '',
      h1: clean(document.querySelector('h1')?.innerText || ''),
      visible_body_text: '',
      blocks: []
    };
  }
  const blocks = [];

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = clean(node.nodeValue || '');
      if (!text || text.length < 2) return NodeFilter.FILTER_REJECT;
      const parent = node.parentElement;
      if (!parent || !isVisible(parent)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });

  let node;
  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    const semantic = parent.closest('h1,h2,h3,h4,p,li,td,th,button') || parent;
    if (!semantic || !isVisible(semantic)) continue;
    const tag = semantic.tagName.toLowerCase();
    const text = tag === 'button' ? clean(semantic.innerText || node.nodeValue || '') : clean(node.nodeValue || '');
    if (!text || text.length < 2) continue;
    const rect = semantic.getBoundingClientRect();
    blocks.push({
      tag,
      text,
      id: semantic.id || '',
      className: typeof semantic.className === 'string' ? semantic.className : '',
      top: Math.round(rect.top + window.scrollY),
      left: Math.round(rect.left + window.scrollX)
    });
  }

  return {
    title: document.title || '',
    url: location.href,
    canonical_url: document.querySelector('link[rel="canonical"]')?.href || '',
    meta_description: document.querySelector('meta[name="description"]')?.content || '',
    h1: clean(document.querySelector('h1')?.innerText || ''),
    root_is_complete_trek_information: true,
    visible_body_text: clean(root.innerText || ''),
    blocks
  };
}
"""


def sections_from_blocks(trek_id: str, page_data: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    seen_block_text: set[str] = set()
    in_scope = bool(page_data.get("root_is_complete_trek_information"))

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = normalize_text("\n".join(current.pop("_lines")))
        if len(text) < 80:
            current = None
            return
        current["text"] = text
        current["char_count"] = len(text)
        current["text_hash"] = text_hash(text)
        sections.append(current)
        current = None

    for block in sorted(page_data.get("blocks") or [], key=lambda item: item.get("top", 0)):
        text = normalize_text(block.get("text", ""))
        if not text:
            continue

        tag = block.get("tag")
        starts_scope = bool(COMPLETE_TREK_INFORMATION_PATTERN.search(text))
        ends_scope = in_scope and tag in {"h1", "h2", "h3", "h4"} and bool(END_OF_COMPLETE_INFORMATION_PATTERN.search(text))
        if ends_scope:
            flush()
            break
        if starts_scope:
            in_scope = True
        if not in_scope:
            continue

        duplicate_key = f"{block.get('tag')}::{block.get('top')}::{block.get('left')}::{text}"
        if duplicate_key in seen_block_text:
            continue
        seen_block_text.add(duplicate_key)

        is_button_heading = tag == "button" and EXPANDABLE_BUTTON_PATTERN.search(text)
        if tag in {"h1", "h2", "h3", "h4"} or is_button_heading:
            heading = re.sub(EXPANDABLE_BUTTON_PATTERN, "", text).strip() or text
            if tag == "button" and heading.lower() == "toggle accordion":
                continue
            flush()
            section_index = len(sections) + 1
            current = {
                "section_id": f"{trek_id}::visible-{section_index:03d}-{stable_id(heading)[:60]}",
                "trek_id": trek_id,
                "source_type": "rendered_visible_dom",
                "content_scope": CONTENT_SCOPE,
                "content_zone": content_zone_for_heading(heading),
                "heading": heading,
                "heading_level": int(tag[1]) if tag.startswith("h") else 2,
                "order": section_index,
                "_lines": [heading],
            }
            continue

        if current is None:
            current = {
                "section_id": f"{trek_id}::visible-001-page-introduction",
                "trek_id": trek_id,
                "source_type": "rendered_visible_dom",
                "content_scope": CONTENT_SCOPE,
                "content_zone": content_zone_for_heading(page_data.get("h1") or "Page introduction"),
                "heading": page_data.get("h1") or "Page introduction",
                "heading_level": 1,
                "order": 1,
                "_lines": [],
            }
        current["_lines"].append(text)

    flush()
    return sections


def scoped_text_from_sections(sections: list[dict[str, Any]]) -> str:
    return normalize_text("\n".join(section["text"] for section in sections))


def page_text_from_sections(sections: list[dict[str, Any]]) -> str:
    parts = []
    for section in sections:
        parts.append(f"# {section['heading']}\n{section['text']}")
    return "\n\n".join(parts) + ("\n" if parts else "")


def scroll_page(page: Any) -> None:
    page.evaluate(
        """
        async () => {
          for (let y = 0; y < document.body.scrollHeight; y += Math.max(600, window.innerHeight)) {
            window.scrollTo(0, y);
            await new Promise(resolve => setTimeout(resolve, 80));
          }
          window.scrollTo(0, 0);
        }
        """
    )


def extract_visible_page_data(page: Any) -> dict[str, Any]:
    return page.evaluate(visible_extraction_script())


def has_complete_trek_information_anchor(page: Any) -> bool:
    return bool(page.locator("#complete-trek-information").count())


def merge_page_data_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if not snapshots:
        return {}
    merged = dict(snapshots[-1])
    blocks = []
    seen: set[tuple[str, str]] = set()
    for snapshot in snapshots:
        for block in snapshot.get("blocks") or []:
            key = (block.get("tag", ""), normalize_text(block.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            blocks.append(block)
    blocks.sort(key=lambda item: (item.get("top", 0), item.get("left", 0), item.get("text", "")))
    merged["blocks"] = blocks
    return merged


def collect_user_accessible_page_states(page: Any) -> tuple[dict[str, Any], int]:
    """Capture visible text after opening each accessible accordion/read-more control."""

    clicked = 0
    clicked_keys: set[str] = set()
    snapshots: list[dict[str, Any]] = []
    scroll_page(page)
    snapshots.append(extract_visible_page_data(page))
    for _ in range(4):
        scroll_page(page)
        buttons = page.locator("button").all()
        round_clicked = 0
        for button in buttons:
            try:
                text = normalize_text(button.inner_text(timeout=500))
                if not EXPANDABLE_BUTTON_PATTERN.search(text):
                    continue
                if not button.is_visible(timeout=500):
                    continue
                box = button.bounding_box(timeout=500) or {}
                key = f"{text[:160]}::{round(box.get('x', 0))}::{round(box.get('y', 0))}"
                if key in clicked_keys:
                    continue
                button.click(timeout=1000, force=True)
                page.wait_for_timeout(120)
                snapshots.append(extract_visible_page_data(page))
                clicked_keys.add(key)
                clicked += 1
                round_clicked += 1
            except Exception:
                continue
        if round_clicked == 0:
            break
    scroll_page(page)
    snapshots.append(extract_visible_page_data(page))
    return merge_page_data_snapshots(snapshots), clicked


def page_diagnostics(page: Any) -> dict[str, str]:
    try:
        return page.evaluate(
            """
            () => ({
              title: document.title || '',
              h1: (document.querySelector('h1')?.innerText || '').replace(/\\s+/g, ' ').trim(),
              body_start: (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 240),
              url: location.href
            })
            """
        )
    except Exception:
        return {"title": "", "h1": "", "body_start": "", "url": ""}


def wait_for_required_anchor(page: Any, retries: int, delay_seconds: float) -> None:
    for attempt in range(retries + 1):
        if has_complete_trek_information_anchor(page):
            return
        if attempt < retries:
            page.wait_for_timeout(int(delay_seconds * 1000))
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass
    details = page_diagnostics(page)
    raise MissingCompleteTrekInformationAnchor(
        "#complete-trek-information missing; "
        f"title={details.get('title')!r}; h1={details.get('h1')!r}; url={details.get('url')!r}"
    )


def scrape_trek(page: Any, trek: dict[str, Any], timeout_ms: int, anchor_retries: int, anchor_retry_delay: float) -> dict[str, Any]:
    url = trek["url"]
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15000))
    except Exception:
        pass
    wait_for_required_anchor(page, anchor_retries, anchor_retry_delay)
    page_data, expanded_controls = collect_user_accessible_page_states(page)
    if page_data.get("missing_complete_trek_information_anchor"):
        raise MissingCompleteTrekInformationAnchor(f"{url} does not have #complete-trek-information")
    sections = sections_from_blocks(trek["uid"], page_data)
    if not sections:
        raise EmptyScopedExtraction(f"{url} has #complete-trek-information but yielded no scoped sections")
    visible_text = normalize_text(page_data.get("visible_body_text", ""))
    return {
        "schema_version": SCHEMA_VERSION,
        "trek_id": trek["uid"],
        "trek_title": trek.get("title", ""),
        "source_url": url,
        "final_url": page_data.get("url", ""),
        "canonical_url": page_data.get("canonical_url", ""),
        "page_title": page_data.get("title", ""),
        "meta_description": page_data.get("meta_description", ""),
        "h1": page_data.get("h1", ""),
        "source_type": "rendered_visible_dom",
        "content_scope": CONTENT_SCOPE,
        "scraped_at": utc_now(),
        "visible_text_hash": text_hash(scoped_text_from_sections(sections)),
        "visible_text_char_count": len(scoped_text_from_sections(sections)),
        "source_container_visible_text_hash": text_hash(visible_text),
        "source_container_visible_text_char_count": len(visible_text),
        "expanded_control_count": expanded_controls,
        "section_count": len(sections),
        "sections": sections,
    }


def write_outputs(out: Path, record: dict[str, Any]) -> None:
    page_path = out / "pages" / f"{record['trek_id']}.json"
    text_path = out / "text" / f"{record['trek_id']}.txt"
    write_json(page_path, record)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(page_text_from_sections(record["sections"]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trek", action="append", help="Trek UID to scrape. Can be passed multiple times.")
    parser.add_argument("--all-active", action="store_true", help="Scrape all active source records instead of the 35 core Himalayan treks.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=45, help="Navigation timeout in seconds.")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--anchor-retries", type=int, default=2)
    parser.add_argument("--anchor-retry-delay", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    treks = select_treks(args)
    print(f"Selected {len(treks)} trek page(s)")
    for trek in treks:
        print(f"- {trek['uid']} {trek['url']}")
    if args.dry_run:
        return

    sync_playwright = import_playwright()
    manifest: list[dict[str, Any]] = []
    started_at = utc_now()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for index, trek in enumerate(treks, start=1):
            page_path = args.out / "pages" / f"{trek['uid']}.json"
            if page_path.exists() and not args.force:
                existing = read_json(page_path)
                manifest.append(
                    {
                        "trek_id": trek["uid"],
                        "url": trek["url"],
                        "status": "skipped_existing",
                        "section_count": existing.get("section_count", 0),
                        "page_file": str(page_path),
                    }
                )
                print(f"[{index}/{len(treks)}] skipped existing {trek['uid']}", flush=True)
                continue

            page = None
            context = None
            try:
                context = browser.new_context(
                    viewport={"width": 1440, "height": 1200},
                    user_agent=USER_AGENT,
                    locale="en-IN",
                )
                page = context.new_page()
                record = scrape_trek(page, trek, args.timeout * 1000, args.anchor_retries, args.anchor_retry_delay)
                page.close()
                context.close()
                write_outputs(args.out, record)
                manifest.append(
                    {
                        "trek_id": trek["uid"],
                        "url": trek["url"],
                        "status": "ok",
                        "section_count": record["section_count"],
                        "visible_text_char_count": record["visible_text_char_count"],
                        "page_file": str(page_path),
                    }
                )
                print(f"[{index}/{len(treks)}] ok {trek['uid']} sections={record['section_count']}", flush=True)
            except MissingCompleteTrekInformationAnchor as exc:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                if args.force:
                    stale_page = args.out / "pages" / f"{trek['uid']}.json"
                    stale_text = args.out / "text" / f"{trek['uid']}.txt"
                    for stale_path in [stale_page, stale_text]:
                        if stale_path.exists():
                            stale_path.unlink()
                manifest.append(
                    {
                        "trek_id": trek["uid"],
                        "url": trek["url"],
                        "status": "skipped_missing_anchor",
                        "error": str(exc),
                    }
                )
                print(f"[{index}/{len(treks)}] skipped missing anchor {trek['uid']}", flush=True)
            except EmptyScopedExtraction as exc:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                if args.force:
                    stale_page = args.out / "pages" / f"{trek['uid']}.json"
                    stale_text = args.out / "text" / f"{trek['uid']}.txt"
                    for stale_path in [stale_page, stale_text]:
                        if stale_path.exists():
                            stale_path.unlink()
                manifest.append({"trek_id": trek["uid"], "url": trek["url"], "status": "skipped_empty_scope", "error": str(exc)})
                print(f"[{index}/{len(treks)}] skipped empty scope {trek['uid']}", flush=True)
            except Exception as exc:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                manifest.append({"trek_id": trek["uid"], "url": trek["url"], "status": "error", "error": repr(exc)})
                print(f"[{index}/{len(treks)}] error {trek['uid']}: {exc!r}", flush=True)
            time.sleep(args.delay)
        browser.close()

    write_json(
        args.out / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source": "rendered_visible_dom",
            "started_at": started_at,
            "finished_at": utc_now(),
            "input": str(args.input),
            "output": str(args.out),
            "count": len(manifest),
            "records": manifest,
        },
    )
    print(f"Wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()

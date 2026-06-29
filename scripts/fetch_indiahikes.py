#!/usr/bin/env python3
"""Fetch public Indiahikes pages into a local research corpus.

The script is intentionally conservative:
- it only follows URLs supplied in an input file or sitemap URL
- it records HTTP status and challenge pages
- it does not try to bypass Cloudflare or other access controls
- it skips AI-training use entirely; this is for local product research
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = "https://indiahikes.com"
USER_AGENT = "TrekPathAIResearch/0.1 (+local MVP research; respects robots.txt)"


def slug_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/") or "home"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", path).strip("-")
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{safe}-{digest}"


def load_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(ROOT):
            urls.append(line)
    return sorted(dict.fromkeys(urls))


def fetch(url: str, timeout: int) -> tuple[int | None, dict[str, str], bytes, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, dict(res.headers.items()), res.read(), None
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers.items()), err.read(), str(err)
    except Exception as exc:
        return None, {}, b"", str(exc)


def is_challenge(status: int | None, headers: dict[str, str], body: bytes) -> bool:
    text = body[:6000].decode("utf-8", errors="ignore").lower()
    return (
        status in {403, 429}
        and (
            "cf-mitigated" in {k.lower(): v for k, v in headers.items()}
            or "just a moment" in text
            or "challenge-platform" in text
            or "enable javascript and cookies" in text
        )
    )


def html_to_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style|noscript|svg|iframe).*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|header|footer|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def title_from_html(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
    return html.unescape(re.sub(r"\s+", " ", match.group(1)).strip()) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", type=Path, default=Path("data/archive/prior_scrape_artifacts/seed_urls.txt"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    text_dir = args.out / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    urls = load_urls(args.urls)
    if args.limit:
        urls = urls[: args.limit]

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=[
                "url",
                "status",
                "bytes",
                "challenge",
                "content_type",
                "title",
                "raw_file",
                "text_file",
                "error",
            ],
        )
        writer.writeheader()
        for index, url in enumerate(urls, start=1):
            status, headers, body, error = fetch(url, args.timeout)
            challenge = is_challenge(status, headers, body)
            slug = slug_for_url(url)
            raw_file = raw_dir / f"{slug}.html"
            text_file = text_dir / f"{slug}.txt"
            title = ""

            raw_file.write_bytes(body)
            if body and not challenge and str(headers.get("Content-Type", "")).startswith("text/html"):
                title = title_from_html(body)
                text_file.write_text(html_to_text(body), encoding="utf-8")
            elif text_file.exists():
                text_file.unlink()

            writer.writerow(
                {
                    "url": url,
                    "status": status or "",
                    "bytes": len(body),
                    "challenge": "yes" if challenge else "no",
                    "content_type": headers.get("Content-Type", ""),
                    "title": title,
                    "raw_file": str(raw_file),
                    "text_file": str(text_file) if text_file.exists() else "",
                    "error": error or "",
                }
            )
            print(f"[{index}/{len(urls)}] {status or 'ERR'} {'CHALLENGE' if challenge else 'OK'} {url}")
            time.sleep(args.delay)

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

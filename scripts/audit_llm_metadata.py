#!/usr/bin/env python3
"""Validate generated LLM trek metadata without calling an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_metadata_common import DEFAULT_OUTPUT, MetadataError, validate_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-report", action="store_true", help="Write reports/llm_metadata_audit.md")
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_profiles(metadata_dir: Path) -> list[dict[str, Any]]:
    profile_dir = metadata_dir / "profiles"
    if not profile_dir.exists():
        raise MetadataError(f"No profile directory found at {profile_dir}")
    results = []
    for path in sorted(profile_dir.glob("*.json")):
        profile = load_json(path)
        errors = validate_profile(profile)
        results.append(
            {
                "path": str(path),
                "trek_id": profile.get("trek_id", path.stem),
                "error_count": len(errors),
                "errors": errors,
            }
        )
    return results


def write_report(results: list[dict[str, Any]]) -> None:
    out = Path("reports/llm_metadata_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LLM Metadata Audit",
        "",
        f"- Profiles checked: {len(results)}",
        f"- Profiles with errors: {sum(1 for item in results if item['errors'])}",
        "",
    ]
    for item in results:
        lines.append(f"## {item['trek_id']}")
        if item["errors"]:
            for error in item["errors"]:
                lines.append(f"- {error}")
        else:
            lines.append("- OK")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def run(args: argparse.Namespace) -> None:
    results = audit_profiles(args.metadata_dir)
    for item in results:
        status = "OK" if not item["errors"] else f"{item['error_count']} error(s)"
        print(f"{item['trek_id']}: {status}")
        for error in item["errors"]:
            print(f"  - {error}")
    if args.write_report:
        write_report(results)
    if any(item["errors"] for item in results):
        raise MetadataError("Metadata audit failed.")


def main() -> None:
    try:
        run(build_parser().parse_args())
    except MetadataError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract evidence-backed trek metadata with Fireworks AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_metadata_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    PROMPT_VERSION,
    MetadataError,
    append_jsonl,
    audit_messages,
    audit_schema,
    call_with_retries,
    consolidation_messages,
    env_model,
    file_sha256,
    load_core_himalayan_treks,
    output_paths,
    profile_schema,
    read_json,
    require_api_key,
    section_messages,
    section_records_for_trek,
    section_schema,
    select_treks,
    utc_now,
    validate_profile,
    validate_section_extraction,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trek", help="Extract one trek by uid")
    parser.add_argument("--all-himalayan", action="store_true", help="Extract all 35 unique core Himalayan trek IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without calling Fireworks or writing metadata")
    parser.add_argument("--execute", action="store_true", help="Call Fireworks and write metadata")
    parser.add_argument("--force", action="store_true", help="Re-run completed section/profile/audit outputs")
    parser.add_argument("--skip-audit", action="store_true", help="Skip the LLM audit pass")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--faq-group-size", type=int, default=5)
    return parser


def section_output_path(paths: dict[str, Path], section_id: str) -> Path:
    filename = section_id.split("::", 1)[1] + ".json"
    return paths["sections_dir"] / filename


def write_progress(
    *,
    args: argparse.Namespace,
    model: str,
    selected: list[dict[str, Any]],
    current_trek: str | None = None,
    current_stage: str | None = None,
    current_section: str | None = None,
    failures: list[dict[str, str]] | None = None,
) -> None:
    sections_done = len(list((args.out / "sections").glob("*/*.json"))) if (args.out / "sections").exists() else 0
    profiles_done = len(list((args.out / "profiles").glob("*.json"))) if (args.out / "profiles").exists() else 0
    audits_done = len(list((args.out / "audits").glob("*.json"))) if (args.out / "audits").exists() else 0
    write_json(
        args.out / "progress.json",
        {
            "schema_version": "v1",
            "updated_at": utc_now(),
            "model": model,
            "selected_trek_count": len(selected),
            "selected_trek_ids": [trek["uid"] for trek in selected],
            "current_trek": current_trek,
            "current_stage": current_stage,
            "current_section": current_section,
            "sections_done": sections_done,
            "profiles_done": profiles_done,
            "audits_done": audits_done,
            "failures": failures or [],
        },
    )


def dry_run(treks: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print(f"Input: {args.input}")
    print(f"Core Himalayan treks: {len(load_core_himalayan_treks(args.input))}")
    print(f"Selected treks: {len(treks)}")
    for trek in treks:
        sections = section_records_for_trek(trek, faq_group_size=args.faq_group_size)
        print(f"- {trek['uid']}: {trek['title']} | sections={len(sections)} | url={trek['url']}")


def extract_section(
    *,
    section: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    model: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    path = section_output_path(paths, section["section_id"])
    if path.exists() and not args.force:
        print(f"  skip section {section['section_id']} (already complete)", flush=True)
        return read_json(path)

    print(f"  extract section {section['section_id']} ({section['section_type']}, {len(section['text'])} chars)", flush=True)
    messages = section_messages(section)
    started_at = utc_now()
    try:
        parsed, raw, attempts = call_with_retries(
            api_key=api_key,
            model=model,
            messages=messages,
            schema_name="SectionExtraction",
            schema=section_schema(),
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )
    except Exception as exc:
        append_jsonl(
            paths["raw"],
            {
                "stage": "section",
                "trek_id": section["trek_id"],
                "section_id": section["section_id"],
                "started_at": started_at,
                "failed_at": utc_now(),
                "error": repr(exc),
                "request": {"model": model, "messages": messages},
            },
        )
        raise

    append_jsonl(
        paths["raw"],
        {
            "stage": "section",
            "trek_id": section["trek_id"],
            "section_id": section["section_id"],
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempts": attempts,
            "request": {"model": model, "messages": messages},
            "response": raw,
        },
    )

    errors = validate_section_extraction(parsed)
    if errors:
        raise MetadataError(f"Schema-invalid section output for {section['section_id']}: {errors}")

    write_json(path, parsed)
    append_jsonl(paths["section_jsonl"], parsed)
    return parsed


def consolidate_trek(
    *,
    trek: dict[str, Any],
    section_outputs: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
    model: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    if paths["profile"].exists() and not args.force:
        print("  skip consolidation (already complete)", flush=True)
        return read_json(paths["profile"])

    print("  consolidate trek metadata", flush=True)
    messages = consolidation_messages(trek, section_outputs)
    started_at = utc_now()
    parsed, raw, attempts = call_with_retries(
        api_key=api_key,
        model=model,
        messages=messages,
        schema_name="TrekMetadataProfile",
        schema=profile_schema(),
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )
    append_jsonl(
        paths["raw"],
        {
            "stage": "consolidation",
            "trek_id": trek["uid"],
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempts": attempts,
            "request": {"model": model, "messages": messages},
            "response": raw,
        },
    )

    errors = validate_profile(parsed)
    if errors:
        raise MetadataError(f"Schema-invalid consolidated profile for {trek['uid']}: {errors}")
    write_json(paths["profile"], parsed)
    return parsed


def audit_trek(
    *,
    trek: dict[str, Any],
    profile: dict[str, Any],
    sections: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str,
    model: str,
    paths: dict[str, Path],
) -> dict[str, Any] | None:
    if args.skip_audit:
        return None
    if paths["audit"].exists() and not args.force:
        print("  skip audit (already complete)", flush=True)
        return read_json(paths["audit"])

    print("  audit trek metadata", flush=True)
    messages = audit_messages(trek, profile, sections)
    started_at = utc_now()
    parsed, raw, attempts = call_with_retries(
        api_key=api_key,
        model=model,
        messages=messages,
        schema_name="MetadataAuditResult",
        schema=audit_schema(),
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )
    append_jsonl(
        paths["raw"],
        {
            "stage": "audit",
            "trek_id": trek["uid"],
            "started_at": started_at,
            "completed_at": utc_now(),
            "attempts": attempts,
            "request": {"model": model, "messages": messages},
            "response": raw,
        },
    )
    write_json(paths["audit"], parsed)
    return parsed


def update_manifest(args: argparse.Namespace, model: str, selected: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
    profiles = sorted(str(path.relative_to(args.out)) for path in (args.out / "profiles").glob("*.json")) if (args.out / "profiles").exists() else []
    audits = sorted(str(path.relative_to(args.out)) for path in (args.out / "audits").glob("*.json")) if (args.out / "audits").exists() else []
    manifest = {
        "schema_version": "v1",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "input_path": str(args.input),
        "input_sha256": file_sha256(args.input),
        "updated_at": utc_now(),
        "selected_trek_count": len(selected),
        "selected_trek_ids": [trek["uid"] for trek in selected],
        "profile_count": len(profiles),
        "audit_count": len(audits),
        "profiles": profiles,
        "audits": audits,
        "failures": failures,
    }
    write_json(args.out / "run_manifest.json", manifest)


def run(args: argparse.Namespace) -> None:
    if args.execute == args.dry_run:
        raise MetadataError("Choose exactly one of --dry-run or --execute.")
    selected = select_treks(args)
    if args.dry_run:
        dry_run(selected, args)
        return

    api_key = require_api_key()
    model = env_model()
    failures: list[dict[str, str]] = []
    write_progress(args=args, model=model, selected=selected, failures=failures)

    for index, trek in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {trek['uid']} - {trek.get('title', '')}", flush=True)
        paths = output_paths(args.out, trek["uid"])
        sections = section_records_for_trek(trek, faq_group_size=args.faq_group_size)
        try:
            section_outputs = []
            for section in sections:
                write_progress(
                    args=args,
                    model=model,
                    selected=selected,
                    current_trek=trek["uid"],
                    current_stage="section",
                    current_section=section["section_id"],
                    failures=failures,
                )
                section_outputs.append(extract_section(section=section, args=args, api_key=api_key, model=model, paths=paths))
            write_progress(args=args, model=model, selected=selected, current_trek=trek["uid"], current_stage="consolidation", failures=failures)
            profile = consolidate_trek(trek=trek, section_outputs=section_outputs, args=args, api_key=api_key, model=model, paths=paths)
            write_progress(args=args, model=model, selected=selected, current_trek=trek["uid"], current_stage="audit", failures=failures)
            audit_trek(trek=trek, profile=profile, sections=sections, args=args, api_key=api_key, model=model, paths=paths)
        except Exception as exc:
            failures.append({"trek_id": trek["uid"], "error": repr(exc)})
            print(f"  FAILED: {exc}", flush=True)
            if args.trek:
                raise
        finally:
            write_progress(args=args, model=model, selected=selected, current_trek=trek["uid"], current_stage="done", failures=failures)
            update_manifest(args, model, selected, failures)

    if failures:
        raise MetadataError(f"{len(failures)} trek(s) failed; see {args.out / 'run_manifest.json'}")
    print(f"Wrote metadata under {args.out}")


def main() -> None:
    try:
        run(build_parser().parse_args())
    except MetadataError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

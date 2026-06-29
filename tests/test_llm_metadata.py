from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ARCHIVED_SCRAPE = ROOT / "data/archive/prior_scrape_artifacts"

from llm_extract_trek_metadata import extract_section, section_output_path
from llm_metadata_common import (
    CANONICAL_URL_BY_UID,
    NON_HIMALAYAN_UIDS,
    load_core_himalayan_treks,
    output_paths,
    read_json,
    section_records_for_trek,
    validate_profile,
    write_json,
)
from scrape_visible_indiahikes import sections_from_blocks, select_treks as select_visible_treks
from build_slim_trek_meta import build_outputs, normalize_quick_facts
from extract_decision_meta import build_embedding_chunks, build_filter_record, consolidate_profile, extraction_targets
from build_trek_knowledge_catalog import build_catalog


class LlmMetadataTests(unittest.TestCase):
    def test_core_himalayan_filter_returns_35_unique_treks(self) -> None:
        treks = load_core_himalayan_treks(ARCHIVED_SCRAPE / "full/trek_facts.json")
        uids = [trek["uid"] for trek in treks]
        self.assertEqual(len(treks), 35)
        self.assertEqual(len(uids), len(set(uids)))
        self.assertNotIn("himalayan-trekking-summer-camps-indiahikes", uids)
        self.assertTrue(NON_HIMALAYAN_UIDS.isdisjoint(uids))
        for uid, url in CANONICAL_URL_BY_UID.items():
            matching = [trek for trek in treks if trek["uid"] == uid]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["url"], url)

    def test_section_records_use_source_sections_without_raw_html(self) -> None:
        trek = next(trek for trek in load_core_himalayan_treks(ARCHIVED_SCRAPE / "full/trek_facts.json") if trek["uid"] == "dayara-bugyal-trek")
        sections = section_records_for_trek(trek)
        section_types = {section["section_type"] for section in sections}
        self.assertIn("quick_info", section_types)
        self.assertIn("difficulty", section_types)
        self.assertIn("best_time", section_types)
        self.assertIn("fitness", section_types)
        self.assertIn("itinerary_day", section_types)
        self.assertIn("faq_group", section_types)
        for section in sections:
            self.assertNotIn("<script", section["text"].lower())
            self.assertEqual(section["source_url"], trek["url"])

    def test_profile_validator_rejects_supported_critical_field_without_evidence(self) -> None:
        profile = {
            "schema_version": "v1",
            "prompt_version": "trek_metadata_v1",
            "metadata_version": "v1",
            "trek_id": "example",
            "source_url": "https://example.com",
            "review_status": "unreviewed",
            "identity": {},
            "geography_and_logistics": {},
            "difficulty": {
                "difficulty_label": {
                    "value": "Easy",
                    "status": "supported",
                    "confidence": "high",
                    "evidence": [],
                }
            },
            "fitness": {},
            "age_and_group_suitability": {},
            "seasonality": {},
            "experience_themes": [],
            "risks_and_watchouts": [],
            "itinerary_profile": {},
            "recommendation_profile": {},
            "comparison_axes": {},
            "needs_review": [],
        }
        errors = validate_profile(profile)
        self.assertTrue(any("difficulty.difficulty_label: missing evidence" in error for error in errors))
        self.assertTrue(any("fitness: missing category output" in error for error in errors))

    def test_completed_section_output_is_reused_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            section = {
                "section_id": "example::quick_info",
                "trek_id": "example",
                "trek_title": "Example",
                "source_url": "https://example.com",
                "section_type": "quick_info",
                "source_field": "quick_info",
                "title": "Quick info",
                "text": "{}",
            }
            paths = output_paths(out, "example")
            expected = {
                "schema_version": "v1",
                "prompt_version": "trek_metadata_v1",
                "trek_id": "example",
                "section_id": "example::quick_info",
                "section_type": "quick_info",
                "source_url": "https://example.com",
                "extracted_items": [],
                "needs_review": [],
            }
            write_json(section_output_path(paths, section["section_id"]), expected)
            args = argparse.Namespace(force=False, temperature=0, timeout=1, retries=0)
            actual = extract_section(section=section, args=args, api_key="", model="", paths=paths)
            self.assertEqual(actual, expected)

    def test_visible_scraper_defaults_to_core_himalayan_treks(self) -> None:
        args = argparse.Namespace(
            input=ARCHIVED_SCRAPE / "full/trek_facts.json",
            all_active=False,
            trek=None,
            limit=0,
        )
        treks = select_visible_treks(args)
        self.assertEqual(len(treks), 35)
        self.assertTrue(all(trek["uid"] not in NON_HIMALAYAN_UIDS for trek in treks))

    def test_visible_sections_are_grouped_by_rendered_headings(self) -> None:
        page_data = {
            "h1": "Example Trek",
            "blocks": [
                {"tag": "h1", "text": "Example Trek", "top": 0},
                {"tag": "h2", "text": "Hero content that must not be scraped", "top": 50},
                {"tag": "p", "text": "This text belongs above the source scope.", "top": 60},
                {"tag": "h2", "text": "Example Trek - Complete Trek Information", "top": 90},
                {"tag": "h2", "text": "Getting Fit", "top": 100},
                {"tag": "p", "text": "You must be able to complete a 5 km run comfortably before this trek.", "top": 120},
                {"tag": "p", "text": "Build cardio and leg strength consistently for several weeks.", "top": 160},
                {"tag": "h2", "text": "What To Pack", "top": 300},
                {"tag": "p", "text": "Carry layers, rain protection, medicines, and required documents.", "top": 320},
                {"tag": "p", "text": "This section should remain visible-source evidence only.", "top": 360},
                {"tag": "h2", "text": "The Indiahikes Spirit of Trekking", "top": 500},
                {"tag": "p", "text": "This footer-like content must not be scraped.", "top": 520},
            ],
        }
        sections = sections_from_blocks("example-trek", page_data)
        self.assertEqual([section["heading"] for section in sections], ["Getting Fit", "What To Pack"])
        self.assertTrue(all(section["source_type"] == "rendered_visible_dom" for section in sections))
        self.assertTrue(all(section["content_scope"] == "complete_trek_information" for section in sections))
        self.assertNotIn("Hero content", "\n".join(section["text"] for section in sections))
        self.assertIn("5 km run", sections[0]["text"])
        self.assertIn("required documents", sections[1]["text"])

    def test_slim_trek_meta_excludes_packing_and_writes_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metas = build_outputs(ROOT / "data/source_packets", out, trek="dayara-bugyal-trek")
            self.assertEqual(len(metas), 1)

            meta_path = out / "dayara-bugyal-trek.meta.json"
            md_path = out / "dayara-bugyal-trek.meta.md"
            manifest_path = out / "manifest.json"
            jsonl_path = out / "all_treks_meta.jsonl"
            self.assertTrue(meta_path.exists())
            self.assertTrue(md_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(jsonl_path.exists())

            meta = read_json(meta_path)
            manifest = read_json(manifest_path)
            jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(meta["schema_version"], "slim_trek_meta_v1")
            self.assertEqual(meta["trek_id"], "dayara-bugyal-trek")
            self.assertTrue(meta["quick_facts"])
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(len(jsonl_rows), 1)
            self.assertEqual(jsonl_rows[0]["trek_id"], "dayara-bugyal-trek")
            self.assertFalse(any(section["source_field"] == "packing_section" for section in meta["sections"]))

            allowed_types = {"difficulty", "seasonality", "fitness", "itinerary_day", "faq", "safety"}
            for section in meta["sections"]:
                for key in ["section_id", "source_field", "section_type", "title", "rendered_support_status", "text"]:
                    self.assertIn(key, section)
                self.assertIn(section["section_type"], allowed_types)
                self.assertTrue(section["text"].strip())

    def test_slim_quick_facts_are_normalized(self) -> None:
        facts = normalize_quick_facts(
            {
                "trek_dificulty": "Moderate - Difficult",
                "trek_duration": "7 days",
                "highest_altitude": "9915 ft",
                "age_limit": "11 - 62 years",
                "total_trek_distance": "48.1 kms",
                "pickup_point": "Example Pickup",
                "pickup_time": "8 AM",
                "drop_off_details": "Example Dropoff at 6 PM",
                "crosstrek_gear_rentals": "Available. Rent Here.",
                "offloading": "Not available",
            }
        )

        self.assertEqual(facts["trek_difficulty"], "Moderate-Difficult")
        self.assertEqual(facts["trek_duration"], "7 days / 48.1 km")
        self.assertEqual(facts["highest_altitude"], "9,915 ft")
        self.assertEqual(facts["suitable_for"], "11 to 62 years")
        self.assertEqual(facts["total_trek_distance"], "48.1 km")
        self.assertEqual(facts["pickup_details"], "Example Pickup at 8 AM")
        self.assertNotIn("pickup_point", facts)
        self.assertNotIn("pickup_time", facts)
        self.assertEqual(facts["dropoff_details"], "Example Dropoff at 6 PM")
        self.assertEqual(facts["gear_rentals"], "Available. Rent Here.")
        self.assertEqual(facts["offloading"], "Not Available")

    def test_decision_meta_filter_record_parses_quick_facts(self) -> None:
        meta = read_json(ROOT / "data/slim_meta/dayara-bugyal-trek.meta.json")
        record = build_filter_record(meta)

        self.assertEqual(record["duration_days"], {"status": "known", "value": 6})
        self.assertEqual(record["distance_km"], {"status": "known", "value": 21.0})
        self.assertEqual(record["highest_altitude_ft"], {"status": "known", "value": 11830})
        self.assertEqual(record["min_age"], {"status": "known", "value": 8})
        self.assertEqual(record["max_age"], {"status": "unknown", "value": None})
        self.assertEqual(record["fitness_required_distance_km"], {"status": "known", "value": 5})
        self.assertEqual(record["fitness_required_time_min"], {"status": "known", "value": 40})
        self.assertEqual(record["offloading_available"], {"status": "known", "value": True})
        self.assertEqual(record["cloakroom_available"], {"status": "known", "value": True})

    def test_decision_meta_targets_and_embedding_chunks_are_lean(self) -> None:
        metas = [
            read_json(ROOT / "data/slim_meta/dayara-bugyal-trek.meta.json"),
            read_json(ROOT / "data/slim_meta/pangarchulla-peak-trek.meta.json"),
        ]
        targets = extraction_targets(metas)
        target_types = [target["section_type"] for target in targets]

        self.assertEqual(target_types.count("seasonality"), 2)
        self.assertEqual(target_types.count("difficulty"), 2)
        self.assertEqual(target_types.count("fitness"), 2)
        self.assertNotIn("faq", target_types)
        self.assertNotIn("itinerary_day", target_types)
        self.assertNotIn("safety", target_types)

        chunks = build_embedding_chunks(metas)
        chunk_types = {chunk["section_type"] for chunk in chunks}
        self.assertIn("faq", chunk_types)
        self.assertIn("quick_fact", chunk_types)
        self.assertIn("itinerary_full", chunk_types)
        self.assertIn("seasonality_full", chunk_types)
        self.assertIn("difficulty_full", chunk_types)
        self.assertIn("fitness_full", chunk_types)
        self.assertNotIn("itinerary_day", chunk_types)
        self.assertNotIn("seasonality", chunk_types)
        self.assertTrue(all("text" in chunk and chunk["text"].strip() for chunk in chunks))
        for trek_id in {meta["trek_id"] for meta in metas}:
            itinerary_chunks = [chunk for chunk in chunks if chunk["trek_id"] == trek_id and chunk["section_type"] == "itinerary_full"]
            self.assertEqual(len(itinerary_chunks), 1)
            self.assertIn("Day", itinerary_chunks[0]["text"])

    def test_compact_decision_profile_has_runtime_shape(self) -> None:
        meta = read_json(ROOT / "data/slim_meta/dayara-bugyal-trek.meta.json")
        filter_record = build_filter_record(meta)
        extraction_dir = ROOT / "data/decision_meta/section_extractions/dayara-bugyal-trek"
        if not extraction_dir.exists():
            self.skipTest("decision meta extraction outputs not generated")
        extractions = [read_json(extraction_dir / f"{section_type}.json") for section_type in ["seasonality", "difficulty", "fitness"]]
        profile = consolidate_profile(meta, filter_record, extractions)
        serialized = json.dumps(profile, ensure_ascii=False)

        self.assertEqual(profile["schema_version"], "trek_decision_profile_v2")
        self.assertIn("candidate_profile", profile)
        self.assertNotIn("decision_profile", profile)
        self.assertNotIn("group_suitability", serialized)
        self.assertNotIn("quote_or_summary", serialized)
        self.assertLess(len(serialized), 7000)
        self.assertLessEqual(len(profile["candidate_profile"]["primary_watchouts"]), 6)
        self.assertNotIn("Safety is a shared responsibility", " ".join(profile["candidate_profile"]["primary_watchouts"]))
        self.assertTrue(any("snow" in value.lower() or "meadow" in value.lower() for value in profile["candidate_profile"]["experience_themes"]))

    def test_trek_knowledge_catalog_joins_current_artifacts(self) -> None:
        catalog = build_catalog(ROOT / "data/slim_meta", ROOT / "data/decision_meta")
        self.assertEqual(catalog["schema_version"], "trek_knowledge_catalog_v1")
        self.assertEqual(catalog["trek_count"], 34)
        dayara = next(trek for trek in catalog["treks"] if trek["trek_id"] == "dayara-bugyal-trek")

        self.assertIn("quick_facts", dayara)
        self.assertIn("decision_profile", dayara)
        self.assertIn("sections", dayara)
        self.assertIn("faq", dayara["sections"])
        self.assertIn("itinerary_day", dayara["sections"])
        self.assertIn("deterministic_filter_source", dayara["quick_facts"]["usage"])
        self.assertIn("runtime_candidate_profile", dayara["decision_profile"]["usage"])
        self.assertTrue(dayara["embedding_chunks"]["count"] > 0)
        self.assertTrue(dayara["sections"]["faq"][0]["text"].strip())


if __name__ == "__main__":
    unittest.main()

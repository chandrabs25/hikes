# Indiahikes Trek Metadata Corpus

This repository now keeps the slim, useful trek metadata in the active data tree. The earlier scrape-heavy corpus is preserved under `data/archive/prior_scrape_artifacts/` for audit and regeneration.

## What Is Kept

- Source packets used by the slim metadata builder in `data/source_packets/`
- Normalized slim metadata in `data/slim_meta/`
- Prior scraped artifacts in `data/archive/prior_scrape_artifacts/`
- Source corpus summaries in `reports/` and `reports/full/`

The archive contains the earlier raw downloaded HTML, plain-text conversions, URL inventories, manifests, source-derived trek facts, rendered visible page extractions, rendered structure files, one-off reorganized review artifacts, and legacy partial LLM metadata outputs.

## What Is Not Trusted Here

The repository intentionally no longer contains normalized recommendation profiles, derived suitability, inferred seasonality, risk tags, experience tags, graph edges, ranking logic, GenAI enrichment outputs, SQLite indexes, API code, or frontend code.

`data/slim_meta` is a metadata-ready source layer for LLM extraction and downstream decision metadata. It preserves useful source page sections such as quick facts, difficulty, best time, fitness, itinerary, FAQs, and safety without retaining the bulky scrape corpus in the active tree. It is not itself a recommendation layer.

The archived scrape corpus remains available for audit and regeneration. Rendered visible page extraction is still the preferred source layer for future LLM metadata work because embedded page data can contain stale or hidden text that is not visible on the live page.

## Scripts

```bash
python3 scripts/fetch_indiahikes.py --delay 2
python3 scripts/analyze_indiahikes_corpus.py
python3 scripts/extract_trek_facts.py
```

For the expanded corpus, pass the full manifest paths explicitly:

```bash
python3 scripts/extract_trek_facts.py \
  --manifest data/archive/prior_scrape_artifacts/full/manifest.csv \
  --out-json data/archive/prior_scrape_artifacts/full/trek_facts.json \
  --out-csv data/archive/prior_scrape_artifacts/full/trek_facts.csv
```

## Rendered Visible Page Scrape

Install Playwright once:

```bash
python3 -m pip install playwright
python3 -m playwright install chromium
```

Scrape the 35 core Himalayan trek pages as rendered visible DOM text. The scraper targets the `#complete-trek-information` page container so the output is limited to the trek information section, including user-accessible accordion content.

```bash
python3 scripts/scrape_visible_indiahikes.py --dry-run
python3 scripts/scrape_visible_indiahikes.py
```

Scrape one trek while testing:

```bash
python3 scripts/scrape_visible_indiahikes.py --trek ali-bedni-bugyal-trek --force
```

Generated files are written to:

```text
data/archive/prior_scrape_artifacts/rendered_pages/manifest.json
data/archive/prior_scrape_artifacts/rendered_pages/pages/<trek_id>.json
data/archive/prior_scrape_artifacts/rendered_pages/text/<trek_id>.txt
```

This output should be treated as the canonical evidence source for user-visible content. The embedded `trek_facts` corpus remains a useful backup and audit layer, especially for comparing hidden source data against rendered page text.

## Current Source Extraction

`data/source_packets/` contains the packetized source used by the slim builder.

`data/slim_meta/` contains normalized slim metadata for 34 treks.

The prior `trek_facts` corpora are archived at:

```text
data/archive/prior_scrape_artifacts/trek_facts.json
data/archive/prior_scrape_artifacts/full/trek_facts.json
```

Regenerate the active slim metadata from source packets:

```bash
python3 scripts/build_slim_trek_meta.py --input data/source_packets --out data/slim_meta
```

## Legacy LLM Metadata Extraction

The archived metadata layer uses Fireworks AI to extract evidence-backed trek metadata from the older source-derived trek sections. It is legacy and not part of the current slim-meta build path.

```bash
python3 scripts/llm_extract_trek_metadata.py --dry-run
FIREWORKS_API_KEY=... python3 scripts/llm_extract_trek_metadata.py --trek dayara-bugyal-trek --execute
FIREWORKS_API_KEY=... python3 scripts/llm_extract_trek_metadata.py --all-himalayan --execute
python3 scripts/audit_llm_metadata.py --write-report
```

Set `FIREWORKS_MODEL` to override the default model (`accounts/fireworks/models/minimax-m3`). Generated files are written under `data/archive/prior_scrape_artifacts/llm_metadata/` and remain unreviewed by default.

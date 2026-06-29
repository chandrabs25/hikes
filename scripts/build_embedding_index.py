from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv


DEFAULT_INPUT = Path("data/decision_meta/embedding_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/decision_meta/vector_index")
DEFAULT_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"


class EmbeddingIndexError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_env() -> None:
    load_dotenv(Path("backend/.env"))
    load_dotenv()


def existing_index_matches(out_dir: Path, source_hash: str, model: str) -> bool:
    manifest_path = out_dir / "manifest.json"
    embeddings_path = out_dir / "embeddings.npy"
    chunks_path = out_dir / "chunks.jsonl"
    if not manifest_path.exists() or not embeddings_path.exists() or not chunks_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("source_hash") == source_hash and manifest.get("model") == model


def call_fireworks_embeddings(
    *,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: int,
) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    request = urllib.request.Request(
        FIREWORKS_EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "indiahikes-embedding-index/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingIndexError(f"Fireworks HTTP {exc.code} {exc.reason}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EmbeddingIndexError(str(exc)) from exc

    try:
        data = sorted(raw["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]
    except (KeyError, TypeError) as exc:
        raise EmbeddingIndexError(f"Unexpected embeddings response shape: {raw}") from exc


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_text(chunk: dict[str, Any]) -> str:
    parts = [
        f"Trek: {chunk.get('trek_title', '')}",
        f"Section: {chunk.get('section_type', '')}",
        f"Title: {chunk.get('title', '')}",
        str(chunk.get("text", "")),
    ]
    return "\n".join(part for part in parts if part.strip())


def compact_chunk(row: int, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": row,
        "chunk_id": chunk["chunk_id"],
        "trek_id": chunk["trek_id"],
        "trek_title": chunk.get("trek_title", ""),
        "section_id": chunk.get("section_id", ""),
        "section_type": chunk.get("section_type", ""),
        "title": chunk.get("title", ""),
        "text": chunk.get("text", ""),
        "source_url": chunk.get("source_url", ""),
    }


def build_index(args: argparse.Namespace) -> None:
    load_env()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise EmbeddingIndexError("FIREWORKS_API_KEY is required to build the embedding index.")

    chunks = read_jsonl(args.input)
    source_hash = file_sha256(args.input)
    model = args.model
    if not args.force and existing_index_matches(args.out, source_hash, model):
        print(f"Index is current: {args.out}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    vectors: list[list[float]] = []
    started = time.time()
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        texts = [build_text(chunk) for chunk in batch]
        batch_vectors = call_fireworks_embeddings(
            api_key=api_key,
            model=model,
            texts=texts,
            timeout=args.timeout,
        )
        vectors.extend(batch_vectors)
        print(f"Embedded {len(vectors)}/{len(chunks)} chunks")

    matrix = normalize(np.asarray(vectors, dtype=np.float32)).astype(np.float32)
    metadata = [compact_chunk(index, chunk) for index, chunk in enumerate(chunks)]
    np.save(args.out / "embeddings.npy", matrix)
    write_jsonl(args.out / "chunks.jsonl", metadata)
    write_json(
        args.out / "manifest.json",
        {
            "schema_version": "trek_embedding_index_v1",
            "model": model,
            "source_file": str(args.input),
            "source_hash": source_hash,
            "chunk_count": len(chunks),
            "embedding_dimensions": int(matrix.shape[1]) if matrix.size else 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - started, 2),
        },
    )
    print(f"Wrote vector index to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local vector index for trek retrieval chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("FIREWORKS_EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    try:
        build_index(build_parser().parse_args())
    except EmbeddingIndexError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

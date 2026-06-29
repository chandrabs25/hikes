from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np


FIREWORKS_EMBEDDINGS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"


class EmbeddingProviderError(RuntimeError):
    """Raised when query embedding fails at the provider boundary."""


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


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
            "User-Agent": "indiahikes-rag-api/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingProviderError(f"Fireworks HTTP {exc.code} {exc.reason}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise EmbeddingProviderError(str(exc)) from exc

    try:
        data = sorted(raw["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]
    except (KeyError, TypeError) as exc:
        raise EmbeddingProviderError(f"Unexpected embeddings response shape: {raw}") from exc


class QueryEmbeddingService:
    def __init__(self) -> None:
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.model = os.getenv("FIREWORKS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.timeout = int(os.getenv("TREK_EMBEDDING_TIMEOUT", "60"))

    def embed_query(self, question: str) -> np.ndarray:
        if not self.api_key:
            raise EmbeddingProviderError("Set FIREWORKS_API_KEY to enable retrieval chat.")
        vectors = call_fireworks_embeddings(
            api_key=self.api_key,
            model=self.model,
            texts=[question],
            timeout=self.timeout,
        )
        return normalize_vector(np.asarray(vectors[0], dtype=np.float32))


PROMPT_CHUNK_CHAR_LIMITS = {
    "itinerary_full": 20000,
    "seasonality_full": 6000,
    "difficulty_full": 6000,
    "fitness_full": 5000,
}


def chunk_text_for_prompt(chunk: dict[str, Any], max_chars: int = 1400) -> str:
    max_chars = PROMPT_CHUNK_CHAR_LIMITS.get(str(chunk.get("section_type", "")), max_chars)
    text = str(chunk.get("text", "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."

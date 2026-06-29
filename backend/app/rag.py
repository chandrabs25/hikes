from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from backend.app.embeddings import chunk_text_for_prompt
from backend.app.schemas import (
    CandidateCard,
    OnboardingState,
    RetrievalCitation,
    TrekChatResponse,
)


FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"


class RagLlmError(RuntimeError):
    """Raised when retrieval-grounded answer generation fails."""


def chat_answer_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "suggested_followups"],
        "properties": {
            "answer": {"type": "string"},
            "suggested_followups": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
            },
        },
    }


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trek_retrieval_answer",
            "schema": chat_answer_schema(),
            "strict": True,
        },
    }


def _compact_onboarding(onboarding: OnboardingState | None) -> dict[str, Any] | None:
    if onboarding is None:
        return None
    return onboarding.model_dump(mode="json", exclude_none=True)


def _selected_trek_refs(cards: list[CandidateCard]) -> list[dict[str, str]]:
    return [{"trek_id": card.trek_id, "title": card.title} for card in cards]


def rag_messages(
    *,
    question: str,
    onboarding: OnboardingState | None,
    selected_cards: list[CandidateCard],
    chunks: list[dict[str, Any]],
    user_context: str | None = None,
) -> list[dict[str, str]]:
    context_chunks = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "trek_id": chunk.get("trek_id"),
            "trek_title": chunk.get("trek_title"),
            "section_type": chunk.get("section_type"),
            "title": chunk.get("title"),
            "text": chunk_text_for_prompt(chunk),
        }
        for chunk in chunks
    ]
    return [
        {
            "role": "system",
                "content": (
                    "You are an Indiahikes trek discussion assistant. "
                    "Answer only from the supplied group onboarding, user_context, and retrieved_chunks. "
                    "Use selected_treks only to identify which trek ids and titles are in scope. "
                    "Each retrieved chunk includes trek_id and trek_title; use those fields to compare treks. "
                "Do not use outside knowledge, do not invent availability or prices, and do not discuss treks outside selected_treks. "
                "If the retrieved chunks are insufficient, say what is missing and ask a useful follow-up. "
                "Keep answers practical for deciding between treks. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "answer_user_question_about_selected_treks",
                    "question": question,
                    "group_onboarding": _compact_onboarding(onboarding),
                    "user_context": user_context,
                    "selected_treks": _selected_trek_refs(selected_cards),
                    "retrieved_chunks": context_chunks,
                },
                ensure_ascii=False,
            ),
        },
    ]


def citations_from_chunks(chunks: list[dict[str, Any]]) -> list[RetrievalCitation]:
    return [
        RetrievalCitation(
            chunk_id=str(chunk.get("chunk_id", "")),
            trek_id=str(chunk.get("trek_id", "")),
            trek_title=str(chunk.get("trek_title", "")),
            section_type=str(chunk.get("section_type", "")),
            title=str(chunk.get("title", "")),
            source_url=chunk.get("source_url"),
            score=chunk.get("score"),
        )
        for chunk in chunks
    ]


def _json_from_string(value: str) -> dict[str, Any] | None:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not cleaned.startswith("{"):
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_answer_text(value: Any) -> str:
    text = str(value or "").strip()
    nested = _json_from_string(text)
    if nested is not None and "answer" in nested:
        text = str(nested.get("answer", "")).strip()
    return text.replace("\\n", "\n").strip()


def _normalise_suggested_followups(data: dict[str, Any]) -> list[str]:
    followups = data.get("suggested_followups", [])
    nested = _json_from_string(str(data.get("answer", "")))
    if nested is not None and not followups:
        followups = nested.get("suggested_followups", [])
    if not isinstance(followups, list):
        return []
    return [str(item) for item in followups if str(item).strip()]


def call_fireworks_rag_answer(
    *,
    api_key: str,
    model: str,
    question: str,
    onboarding: OnboardingState | None,
    selected_cards: list[CandidateCard],
    chunks: list[dict[str, Any]],
    user_context: str | None = None,
    temperature: float = 0,
    timeout: int = 120,
) -> TrekChatResponse:
    payload = {
        "model": model,
        "messages": rag_messages(
            question=question,
            onboarding=onboarding,
            user_context=user_context,
            selected_cards=selected_cards,
            chunks=chunks,
        ),
        "temperature": temperature,
        "response_format": response_format(),
    }
    request = urllib.request.Request(
        FIREWORKS_URL,
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
        content = raw["choices"][0]["message"]["content"]
        data = json.loads(content)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RagLlmError(f"Fireworks HTTP {exc.code} {exc.reason}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        raise RagLlmError(str(exc)) from exc

    return TrekChatResponse(
        mode="live",
        answer=_normalise_answer_text(data.get("answer", "")),
        suggested_followups=_normalise_suggested_followups(data),
        citations=citations_from_chunks(chunks),
        used_trek_ids=sorted({str(card.trek_id) for card in selected_cards}),
    )


class RagAnswerService:
    def __init__(self) -> None:
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.model = os.getenv("FIREWORKS_MODEL", DEFAULT_MODEL)
        self.temperature = float(os.getenv("TREK_RAG_TEMPERATURE", "0"))
        self.timeout = int(os.getenv("TREK_RAG_TIMEOUT", "120"))

    def answer(
        self,
        *,
        question: str,
        onboarding: OnboardingState | None,
        selected_cards: list[CandidateCard],
        chunks: list[dict[str, Any]],
        user_context: str | None = None,
    ) -> TrekChatResponse:
        if not self.api_key:
            raise RagLlmError("Set FIREWORKS_API_KEY to enable retrieval chat.")
        return call_fireworks_rag_answer(
            api_key=self.api_key,
            model=self.model,
            question=question,
            onboarding=onboarding,
            user_context=user_context,
            selected_cards=selected_cards,
            chunks=chunks,
            temperature=self.temperature,
            timeout=self.timeout,
        )

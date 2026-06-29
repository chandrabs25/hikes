from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from backend.app.schemas import (
    CandidateCard,
    LlmRecommendation,
    OnboardingState,
    RecommendedTrek,
    ShortlistResponse,
    TrekComparison,
)


FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"


class RecommendationLlmError(RuntimeError):
    """Raised when shortlist reasoning fails at the provider boundary."""


def recommendation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["recommended", "comparison", "questions_to_refine", "notes"],
        "properties": {
            "recommended": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "trek_id",
                        "title",
                        "recommendation",
                        "reasons",
                        "tradeoffs",
                        "person_specific_notes",
                    ],
                    "properties": {
                        "trek_id": {"type": "string"},
                        "title": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "reasons": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                        "tradeoffs": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                        "person_specific_notes": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "comparison": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["trek_id", "title", "best_fit_for", "concerns"],
                    "properties": {
                        "trek_id": {"type": "string"},
                        "title": {"type": "string"},
                        "best_fit_for": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                        "concerns": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                    },
                },
            },
            "questions_to_refine": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
    }


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trek_shortlist_recommendation",
            "schema": recommendation_schema(),
            "strict": True,
        },
    }


def _compact_onboarding(onboarding: OnboardingState) -> dict[str, Any]:
    return onboarding.model_dump(mode="json", exclude_none=True)


def _compact_cards(cards: list[CandidateCard]) -> list[dict[str, Any]]:
    return [card.model_dump(mode="json", exclude_none=True) for card in cards]


def recommendation_messages(
    onboarding: OnboardingState,
    eligible: list[CandidateCard],
    conditional: list[CandidateCard],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a trek recommendation assistant for Indiahikes. "
                "Reason only from the supplied group profile and candidate cards. "
                "The user profile may be lightweight: name, age, optional person notes, optional preferences, and text_input. "
                "Do not invent scores, rankings, prices, availability, or facts not present in the input. "
                "Treat empty candidate fields as unknown, not positive or negative evidence. "
                "Do not claim a trek is family-friendly, quiet, crowded, technical, or safe unless the candidate card supports it. "
                "Recommend only treks from eligible_candidates unless conditional_candidates are clearly a better fit, "
                "and mention tradeoffs plainly. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "choose_best_treks_for_group",
                    "selection_rules": [
                        "Prioritize suitability for the whole group, especially the least-prepared participant.",
                        "Use each participant's notes for person-specific concerns, preferences, medical or fitness conditions.",
                        "Use deterministic filters as already applied; do not reverse exclusions.",
                        "Use text_input as the primary source for subjective goals, concerns, deal breakers, and success criteria.",
                        "Use facts and decision_axes as grounded evidence.",
                        "Put individual implications in person_specific_notes, naming the person when useful.",
                        "Separate direct candidate facts from your own cautious inferences in wording.",
                        "Keep recommendations concise and practical.",
                        "Use questions_to_refine only for missing information that would materially change the shortlist.",
                    ],
                    "group_onboarding": _compact_onboarding(onboarding),
                    "eligible_candidates": _compact_cards(eligible),
                    "conditional_candidates": _compact_cards(conditional),
                },
                ensure_ascii=False,
            ),
        },
    ]


def call_fireworks_recommendation(
    *,
    api_key: str,
    model: str,
    onboarding: OnboardingState,
    eligible: list[CandidateCard],
    conditional: list[CandidateCard],
    temperature: float = 0,
    timeout: int = 120,
) -> LlmRecommendation:
    payload = {
        "model": model,
        "messages": recommendation_messages(onboarding, eligible, conditional),
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
            "User-Agent": "indiahikes-shortlisting-api/1.0",
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
        raise RecommendationLlmError(f"Fireworks HTTP {exc.code} {exc.reason}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        raise RecommendationLlmError(str(exc)) from exc

    return LlmRecommendation(
        mode="live",
        recommended=[RecommendedTrek(**item) for item in data.get("recommended", [])],
        comparison=[TrekComparison(**item) for item in data.get("comparison", [])],
        questions_to_refine=list(data.get("questions_to_refine", [])),
        notes=str(data.get("notes", "")),
    )


class RecommendationService:
    def __init__(self) -> None:
        self.api_key = os.getenv("FIREWORKS_API_KEY")
        self.model = os.getenv("FIREWORKS_MODEL", DEFAULT_MODEL)
        self.temperature = float(os.getenv("TREK_LLM_TEMPERATURE", "0"))
        self.timeout = int(os.getenv("TREK_LLM_TIMEOUT", "120"))

    def recommend(self, onboarding: OnboardingState, shortlist: ShortlistResponse) -> LlmRecommendation:
        if not self.api_key:
            raise RecommendationLlmError("Set FIREWORKS_API_KEY to enable live LLM reasoning.")
        return call_fireworks_recommendation(
            api_key=self.api_key,
            model=self.model,
            onboarding=onboarding,
            eligible=shortlist.eligible_candidates,
            conditional=shortlist.conditional_candidates,
            temperature=self.temperature,
            timeout=self.timeout,
        )

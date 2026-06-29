from __future__ import annotations

import collections
from dataclasses import dataclass

from backend.app.cards import build_candidate_card
from backend.app.data import TrekRecord, known_value, raw_fact
from backend.app.schemas import (
    AppliedFilters,
    ExclusionSummary,
    LlmRecommendation,
    OnboardingState,
    ShortlistResponse,
)


DIFFICULTY_ORDER = {
    "Easy": 1,
    "Easy-Moderate": 2,
    "Moderate": 3,
    "Moderate-Difficult": 4,
    "Difficult": 5,
}
REVERSE_DIFFICULTY = {value: key for key, value in DIFFICULTY_ORDER.items()}
MAX_CANDIDATES = 6
MIN_TARGET_CANDIDATES = 3


@dataclass(frozen=True)
class GroupConstraints:
    youngest_age: int | None
    oldest_age: int | None


def group_constraints(onboarding: OnboardingState) -> GroupConstraints:
    ages = [participant.age for participant in onboarding.participants]
    return GroupConstraints(
        youngest_age=min(ages) if ages else None,
        oldest_age=max(ages) if ages else None,
    )


def difficulty_level(value: str | None) -> int | None:
    if value is None:
        return None
    return DIFFICULTY_ORDER.get(value)


def adjacent_difficulties(target: str | None) -> list[str]:
    level = difficulty_level(target)
    if level is None:
        return []
    nearby = []
    for adjacent_level in (level - 1, level + 1):
        if adjacent_level in REVERSE_DIFFICULTY:
            nearby.append(REVERSE_DIFFICULTY[adjacent_level])
    return nearby


def hard_exclusion_reason(trek: TrekRecord, onboarding: OnboardingState) -> str | None:
    record = trek.filter_record
    prefs = onboarding.preferences
    group = group_constraints(onboarding)

    min_age = known_value(record, "min_age")
    if group.youngest_age is not None and min_age is not None and group.youngest_age < min_age:
        return "youngest_age_below_trek_minimum"

    max_age = known_value(record, "max_age")
    if group.oldest_age is not None and max_age is not None and group.oldest_age > max_age:
        return "oldest_age_above_trek_maximum"

    duration = known_value(record, "duration_days")
    if prefs.duration_days is not None and duration is not None and duration != prefs.duration_days:
        return "duration_mismatch"

    altitude = known_value(record, "highest_altitude_ft")
    if prefs.altitude_ceiling_ft is not None and altitude is not None and altitude > prefs.altitude_ceiling_ft:
        return "altitude_exceeds_ceiling"

    if prefs.needs_offloading is True and known_value(record, "offloading_available") is False:
        return "offloading_required_but_unavailable"

    return None


def deterministic_priority(trek: TrekRecord, onboarding: OnboardingState) -> tuple[int, int, int, int, str]:
    record = trek.filter_record
    prefs = onboarding.preferences

    pickup_match = 0
    if prefs.preferred_pickup_cities:
        pickup = str(known_value(record, "pickup_city") or "").casefold()
        preferred = {city.casefold() for city in prefs.preferred_pickup_cities}
        pickup_match = 1 if pickup in preferred else 0

    offload_match = 1 if prefs.needs_offloading is True and known_value(record, "offloading_available") is True else 0

    duration = known_value(record, "duration_days") or 999
    return (-pickup_match, -offload_match, duration, trek.trek_title.casefold())


def shortlist_treks(treks: list[TrekRecord], onboarding: OnboardingState) -> ShortlistResponse:
    exclusions: collections.Counter[str] = collections.Counter()
    hard_passed: list[TrekRecord] = []
    for trek in treks:
        reason = hard_exclusion_reason(trek, onboarding)
        if reason:
            exclusions[reason] += 1
        else:
            hard_passed.append(trek)

    target = onboarding.preferences.target_difficulty.value if onboarding.preferences.target_difficulty else None
    if target:
        primary = [trek for trek in hard_passed if raw_fact(trek.filter_record, "trek_difficulty") == target]
        exclusions["outside_target_difficulty"] += len(hard_passed) - len(primary)
    else:
        primary = list(hard_passed)

    primary = sorted(primary, key=lambda trek: deterministic_priority(trek, onboarding))
    eligible = primary[:MAX_CANDIDATES]

    conditional: list[TrekRecord] = []
    if target and len(eligible) < MIN_TARGET_CANDIDATES:
        adjacent = set(adjacent_difficulties(target))
        already = {trek.trek_id for trek in eligible}
        conditional = [
            trek
            for trek in hard_passed
            if trek.trek_id not in already and raw_fact(trek.filter_record, "trek_difficulty") in adjacent
        ]
        conditional = sorted(conditional, key=lambda trek: deterministic_priority(trek, onboarding))
        conditional = conditional[: max(0, MIN_TARGET_CANDIDATES - len(eligible))]

    group = group_constraints(onboarding)
    prefs = onboarding.preferences
    return ShortlistResponse(
        applied_filters=AppliedFilters(
            youngest_age=group.youngest_age,
            oldest_age=group.oldest_age,
            target_difficulty=prefs.target_difficulty,
            duration_days=prefs.duration_days,
            altitude_ceiling_ft=prefs.altitude_ceiling_ft,
            preferred_pickup_cities=prefs.preferred_pickup_cities,
            needs_offloading=prefs.needs_offloading,
        ),
        eligible_candidates=[build_candidate_card(trek) for trek in eligible],
        conditional_candidates=[build_candidate_card(trek) for trek in conditional],
        excluded=[
            ExclusionSummary(reason=reason, count=count)
            for reason, count in sorted(exclusions.items())
            if count
        ],
        llm_recommendation=LlmRecommendation(),
    )

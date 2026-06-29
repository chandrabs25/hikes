from __future__ import annotations

import re
from typing import Any

from backend.app.data import TrekRecord, known_value, raw_fact
from backend.app.schemas import (
    AgeRange,
    CandidateCard,
    CandidateFacts,
    DecisionAxes,
    DecisionNotes,
    DiscomfortProfileAxes,
    ExperienceMatchAxes,
    GroupFitAxes,
    LogisticsPoint,
)


MAX_AXIS_ITEMS = 2
MAX_NOTE_ITEMS = 3


AXIS_PATTERNS = {
    "family_or_child": re.compile(r"\bfamil|child|children|kid|introduc", re.I),
    "snow": re.compile(r"\bsnow|snowfall|winter|frozen|ice\b", re.I),
    "meadows": re.compile(r"\bmeadow|bugyal|grassland|pasture\b", re.I),
    "views": re.compile(r"\bview|mountain|range|panorama|photograph|sunrise|sunset|sky|skies\b", re.I),
    "summit_or_adventure": re.compile(r"\bsummit|peak|pass|adventur|ridge|technical|boulder\b", re.I),
    "forests_or_flowers": re.compile(r"\bforest|oak|rhododendron|flower|wildflower|maple|pine|deodar\b", re.I),
    "solitude_or_crowds": re.compile(r"\bcrowd|solitude|empty trail|isolation|quiet|peaceful|less crowded\b", re.I),
    "cold_or_snow": re.compile(r"\bcold|snow|snowfall|winter|frozen|ice|minus|-\\d|warm layer\b", re.I),
    "steep_or_strenuous": re.compile(r"\bsteep|strenuous|climb|ascent|descent|boulder|long day|summit day\b", re.I),
    "altitude": re.compile(r"\baltitude|ams|acute mountain|acclimati|above 10,000|high-altitude\b", re.I),
    "weather_uncertainty": re.compile(r"\brain|monsoon|weather|landslide|roadblock|slush|closed|unsafe\b", re.I),
    "crowds": re.compile(r"\bcrowd|busy|less crowded|solitude|empty trail|isolation|quiet\b", re.I),
}


def _dedupe_limit(values: list[Any], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _matching(values: list[Any], pattern_name: str, limit: int = MAX_AXIS_ITEMS) -> list[str]:
    pattern = AXIS_PATTERNS[pattern_name]
    return _dedupe_limit([value for value in values if pattern.search(str(value))], limit)


def _profile_section(trek: TrekRecord) -> dict[str, Any]:
    return ((trek.profile or {}).get("candidate_profile") or {})


def _fitness_label(record: dict[str, Any]) -> str | None:
    distance = known_value(record, "fitness_required_distance_km")
    minutes = known_value(record, "fitness_required_time_min")
    if distance is None or minutes is None:
        return None
    return f"{distance:g} km in {minutes:g} min"


def _age_labels(record: dict[str, Any]) -> list[str]:
    min_age = known_value(record, "min_age")
    max_age = known_value(record, "max_age")
    labels = []
    if min_age is not None and max_age is not None:
        labels.append(f"Age range {min_age}-{max_age}")
    elif min_age is not None:
        labels.append(f"Min age {min_age}")
    elif max_age is not None:
        labels.append(f"Max age {max_age}")
    return labels


def _all_experience_text(profile: dict[str, Any]) -> list[Any]:
    seasonality = profile.get("seasonality") or {}
    return (
        list(profile.get("experience_themes") or [])
        + list(profile.get("best_for") or [])
        + list(seasonality.get("snow_or_rain_notes") or [])
    )


def _all_discomfort_text(profile: dict[str, Any]) -> list[Any]:
    difficulty = profile.get("difficulty") or {}
    seasonality = profile.get("seasonality") or {}
    return (
        list(profile.get("primary_watchouts") or [])
        + list(profile.get("not_ideal_for") or [])
        + list(difficulty.get("terrain_challenges") or [])
        + list(difficulty.get("altitude_or_weather_risks") or [])
        + list(seasonality.get("snow_or_rain_notes") or [])
    )


def _build_tradeoffs(profile: dict[str, Any]) -> list[str]:
    watchouts = list(profile.get("primary_watchouts") or [])
    not_ideal = list(profile.get("not_ideal_for") or [])
    difficulty = profile.get("difficulty") or {}
    terrain = list(difficulty.get("terrain_challenges") or [])
    return _dedupe_limit(watchouts + not_ideal + terrain, MAX_NOTE_ITEMS)


def build_candidate_card(trek: TrekRecord) -> CandidateCard:
    record = trek.filter_record
    profile = _profile_section(trek)
    difficulty = profile.get("difficulty") or {}
    fitness_label = _fitness_label(record)

    experience_text = _all_experience_text(profile)
    discomfort_text = _all_discomfort_text(profile)

    family_signals = (
        _matching(difficulty.get("experience_suitability") or [], "family_or_child")
        + _matching(profile.get("best_for") or [], "family_or_child")
        + _matching(profile.get("experience_themes") or [], "family_or_child")
    )

    return CandidateCard(
        trek_id=trek.trek_id,
        title=trek.trek_title,
        facts=CandidateFacts(
            difficulty=raw_fact(record, "trek_difficulty"),
            duration_days=known_value(record, "duration_days"),
            distance_km=known_value(record, "distance_km"),
            altitude_ft=known_value(record, "highest_altitude_ft"),
            age_range=AgeRange(min=known_value(record, "min_age"), max=known_value(record, "max_age")),
            fitness=fitness_label,
            pickup=LogisticsPoint(city=known_value(record, "pickup_city"), time=known_value(record, "pickup_time")),
            dropoff=LogisticsPoint(city=known_value(record, "dropoff_city"), time=known_value(record, "dropoff_time")),
            offloading=known_value(record, "offloading_available"),
            cloakroom=known_value(record, "cloakroom_available"),
            accommodation=known_value(record, "accommodation_type"),
        ),
        decision_axes=DecisionAxes(
            group_fit=GroupFitAxes(
                age=_age_labels(record),
                fitness=[fitness_label] if fitness_label else [],
                experience=_dedupe_limit(difficulty.get("experience_suitability") or [], MAX_AXIS_ITEMS),
                family_or_child=_dedupe_limit(family_signals, MAX_AXIS_ITEMS),
            ),
            experience_match=ExperienceMatchAxes(
                snow=_matching(experience_text, "snow"),
                meadows=_matching(experience_text, "meadows"),
                views=_matching(experience_text, "views"),
                summit_or_adventure=_matching(experience_text, "summit_or_adventure"),
                forests_or_flowers=_matching(experience_text, "forests_or_flowers"),
                solitude_or_crowds=_matching(experience_text, "solitude_or_crowds"),
            ),
            discomfort_profile=DiscomfortProfileAxes(
                cold_or_snow=_matching(discomfort_text, "cold_or_snow"),
                steep_or_strenuous=_matching(discomfort_text, "steep_or_strenuous"),
                altitude=_matching(discomfort_text, "altitude"),
                weather_uncertainty=_matching(discomfort_text, "weather_uncertainty"),
                crowds=_matching(discomfort_text, "crowds"),
            ),
            decision_notes=DecisionNotes(
                best_for=_dedupe_limit(profile.get("best_for") or [], MAX_NOTE_ITEMS),
                not_ideal_for=_dedupe_limit(profile.get("not_ideal_for") or [], MAX_NOTE_ITEMS),
                key_tradeoffs=_build_tradeoffs(profile),
            ),
        ),
    )

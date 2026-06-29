from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Difficulty(str, Enum):
    EASY = "Easy"
    EASY_MODERATE = "Easy-Moderate"
    MODERATE = "Moderate"
    MODERATE_DIFFICULT = "Moderate-Difficult"
    DIFFICULT = "Difficult"


class ParticipantProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int = Field(gt=0)
    notes: str | None = None


class TripPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    travel_months: list[str] = Field(default_factory=list)
    target_difficulty: Difficulty | None = None
    duration_days: int | None = Field(default=None, gt=0)
    altitude_ceiling_ft: int | None = Field(default=None, gt=0)
    preferred_pickup_cities: list[str] = Field(default_factory=list)
    needs_offloading: bool | None = None
    themes: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)


class OnboardingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participants: list[ParticipantProfile] = Field(default_factory=list)
    preferences: TripPreferences = Field(default_factory=TripPreferences)
    text_input: str | None = None


class CreateSessionRequest(BaseModel):
    trip_name: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str


class AgeRange(BaseModel):
    min: int | None = None
    max: int | None = None


class LogisticsPoint(BaseModel):
    city: str | None = None
    time: str | None = None


class CandidateFacts(BaseModel):
    difficulty: str | None = None
    duration_days: int | None = None
    distance_km: float | None = None
    altitude_ft: int | None = None
    age_range: AgeRange = Field(default_factory=AgeRange)
    fitness: str | None = None
    pickup: LogisticsPoint = Field(default_factory=LogisticsPoint)
    dropoff: LogisticsPoint = Field(default_factory=LogisticsPoint)
    offloading: bool | None = None
    cloakroom: bool | None = None
    accommodation: str | None = None


class GroupFitAxes(BaseModel):
    age: list[str] = Field(default_factory=list)
    fitness: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    family_or_child: list[str] = Field(default_factory=list)


class ExperienceMatchAxes(BaseModel):
    snow: list[str] = Field(default_factory=list)
    meadows: list[str] = Field(default_factory=list)
    views: list[str] = Field(default_factory=list)
    summit_or_adventure: list[str] = Field(default_factory=list)
    forests_or_flowers: list[str] = Field(default_factory=list)
    solitude_or_crowds: list[str] = Field(default_factory=list)


class DiscomfortProfileAxes(BaseModel):
    cold_or_snow: list[str] = Field(default_factory=list)
    steep_or_strenuous: list[str] = Field(default_factory=list)
    altitude: list[str] = Field(default_factory=list)
    weather_uncertainty: list[str] = Field(default_factory=list)
    crowds: list[str] = Field(default_factory=list)


class DecisionNotes(BaseModel):
    best_for: list[str] = Field(default_factory=list)
    not_ideal_for: list[str] = Field(default_factory=list)
    key_tradeoffs: list[str] = Field(default_factory=list)


class DecisionAxes(BaseModel):
    group_fit: GroupFitAxes = Field(default_factory=GroupFitAxes)
    experience_match: ExperienceMatchAxes = Field(default_factory=ExperienceMatchAxes)
    discomfort_profile: DiscomfortProfileAxes = Field(default_factory=DiscomfortProfileAxes)
    decision_notes: DecisionNotes = Field(default_factory=DecisionNotes)


class CandidateCard(BaseModel):
    trek_id: str
    title: str
    source_url: str | None = None
    image_url: str | None = None
    video_url: str | None = None
    facts: CandidateFacts
    decision_axes: DecisionAxes = Field(default_factory=DecisionAxes)


class ExclusionSummary(BaseModel):
    reason: str
    count: int


class AppliedFilters(BaseModel):
    youngest_age: int | None = None
    oldest_age: int | None = None
    target_difficulty: Difficulty | None = None
    duration_days: int | None = None
    altitude_ceiling_ft: int | None = None
    preferred_pickup_cities: list[str] = Field(default_factory=list)
    needs_offloading: bool | None = None


class RecommendedTrek(BaseModel):
    trek_id: str
    title: str
    recommendation: str
    reasons: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    person_specific_notes: list[str] = Field(default_factory=list)


class TrekComparison(BaseModel):
    trek_id: str
    title: str
    best_fit_for: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class LlmRecommendation(BaseModel):
    mode: str = "live"
    recommended: list[RecommendedTrek] = Field(default_factory=list)
    comparison: list[TrekComparison] = Field(default_factory=list)
    questions_to_refine: list[str] = Field(default_factory=list)
    notes: str = ""


class ShortlistResponse(BaseModel):
    applied_filters: AppliedFilters
    eligible_candidates: list[CandidateCard]
    conditional_candidates: list[CandidateCard]
    excluded: list[ExclusionSummary]
    llm_recommendation: LlmRecommendation


class ComparisonTableRow(BaseModel):
    trek_id: str
    title: str
    image_url: str | None = None
    video_url: str | None = None
    difficulty: str | None = None
    duration_days: int | None = None
    distance_km: float | None = None
    altitude_ft: int | None = None
    age_range: AgeRange = Field(default_factory=AgeRange)
    fitness: str | None = None
    pickup: LogisticsPoint = Field(default_factory=LogisticsPoint)
    dropoff: LogisticsPoint = Field(default_factory=LogisticsPoint)
    offloading: bool | None = None
    cloakroom: bool | None = None
    accommodation: str | None = None


class ComparisonTableResponse(BaseModel):
    session_id: str
    source: str = "latest_llm_recommendations"
    rows: list[ComparisonTableRow]


class TrekChatRequest(BaseModel):
    question: str = Field(min_length=1)
    trek_ids: list[str] = Field(default_factory=list)
    section_types: list[str] = Field(default_factory=list)
    max_chunks: int = Field(default=8, ge=1, le=12)


class RetrievalCitation(BaseModel):
    chunk_id: str
    trek_id: str
    trek_title: str
    section_type: str
    title: str
    source_url: str | None = None
    score: float | None = None


class TrekChatResponse(BaseModel):
    mode: str = "live"
    answer: str
    suggested_followups: list[str] = Field(default_factory=list)
    citations: list[RetrievalCitation] = Field(default_factory=list)
    used_trek_ids: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str
    trip_name: str | None = None
    onboarding: OnboardingState | None = None
    latest_shortlist: ShortlistResponse | None = None


class HealthResponse(BaseModel):
    status: str
    trek_count: int


class RangeOption(BaseModel):
    min: int | float | None = None
    max: int | float | None = None


class FilterOptionsResponse(BaseModel):
    difficulty_buckets: list[str]
    pickup_cities: list[str]
    duration_days: RangeOption
    altitude_ft: RangeOption
    age: RangeOption
    fitness_required_distance_km: RangeOption
    fitness_required_time_min: RangeOption

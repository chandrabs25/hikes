from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.app.data import TrekRepository
from backend.app.filtering import shortlist_treks
from backend.app.llm import RecommendationService
from backend.app.llm import RecommendationLlmError
from backend.app.schemas import (
    ComparisonTableResponse,
    ComparisonTableRow,
    CreateSessionRequest,
    CreateSessionResponse,
    FilterOptionsResponse,
    HealthResponse,
    OnboardingState,
    SessionState,
    ShortlistResponse,
)
from backend.app.sessions import SessionStore


repository = TrekRepository()
session_store = SessionStore()
recommendation_service = RecommendationService()

app = FastAPI(title="Indiahikes Trek Shortlisting API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", trek_count=len(repository.treks))


@app.get("/treks/filter-options", response_model=FilterOptionsResponse)
def filter_options() -> FilterOptionsResponse:
    return FilterOptionsResponse(**repository.filter_options())


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    session = session_store.create(trip_name=request.trip_name)
    return CreateSessionResponse(session_id=session.session_id)


@app.put("/sessions/{session_id}/onboarding", response_model=SessionState)
def update_onboarding(session_id: str, onboarding: OnboardingState) -> SessionState:
    session = session_store.set_onboarding(session_id, onboarding)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/sessions/{session_id}/shortlist", response_model=ShortlistResponse)
def create_shortlist(session_id: str) -> ShortlistResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.onboarding is None:
        raise HTTPException(status_code=400, detail="Onboarding must be completed before shortlisting")
    shortlist = shortlist_treks(repository.treks, session.onboarding)
    try:
        shortlist.llm_recommendation = recommendation_service.recommend(session.onboarding, shortlist)
    except RecommendationLlmError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    session_store.set_shortlist(session_id, shortlist)
    return shortlist


@app.get("/sessions/{session_id}/comparison-table", response_model=ComparisonTableResponse)
def get_comparison_table(session_id: str) -> ComparisonTableResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.latest_shortlist is None:
        raise HTTPException(status_code=400, detail="Run shortlist before requesting comparison table")

    cards_by_id = {
        card.trek_id: card
        for card in session.latest_shortlist.eligible_candidates
        + session.latest_shortlist.conditional_candidates
    }
    rows: list[ComparisonTableRow] = []
    for recommendation in session.latest_shortlist.llm_recommendation.recommended:
        card = cards_by_id.get(recommendation.trek_id)
        if card is None:
            continue
        facts = card.facts
        rows.append(
            ComparisonTableRow(
                trek_id=card.trek_id,
                title=card.title,
                difficulty=facts.difficulty,
                duration_days=facts.duration_days,
                distance_km=facts.distance_km,
                altitude_ft=facts.altitude_ft,
                age_range=facts.age_range,
                fitness=facts.fitness,
                pickup=facts.pickup,
                dropoff=facts.dropoff,
                offloading=facts.offloading,
                cloakroom=facts.cloakroom,
                accommodation=facts.accommodation,
            )
        )
    return ComparisonTableResponse(session_id=session_id, rows=rows)


@app.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str) -> SessionState:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

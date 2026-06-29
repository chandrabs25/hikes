from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import cors_origins, load_backend_env
from backend.app.data import TrekRepository
from backend.app.embeddings import EmbeddingProviderError, QueryEmbeddingService
from backend.app.filtering import shortlist_treks
from backend.app.llm import RecommendationService
from backend.app.llm import RecommendationLlmError
from backend.app.rag import RagAnswerService, RagLlmError
from backend.app.retrieval import RetrievalIndexError, TrekVectorIndex
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
    TrekChatRequest,
    TrekChatResponse,
)
from backend.app.sessions import SessionStore


load_backend_env()
repository = TrekRepository()
session_store = SessionStore()
recommendation_service = RecommendationService()
query_embedding_service = QueryEmbeddingService()
vector_index = TrekVectorIndex()
rag_answer_service = RagAnswerService()

app = FastAPI(title="Indiahikes Trek Shortlisting API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
                image_url=card.image_url,
                video_url=card.video_url,
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


@app.post("/sessions/{session_id}/chat", response_model=TrekChatResponse)
def chat_about_treks(session_id: str, request: TrekChatRequest) -> TrekChatResponse:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.latest_shortlist is None:
        raise HTTPException(status_code=400, detail="Run shortlist before starting trek chat")
    if not vector_index.available:
        raise HTTPException(status_code=503, detail="Build data/decision_meta/vector_index before using retrieval chat")

    cards_by_id = {
        card.trek_id: card
        for card in session.latest_shortlist.eligible_candidates
        + session.latest_shortlist.conditional_candidates
    }
    recommended_ids = [item.trek_id for item in session.latest_shortlist.llm_recommendation.recommended]
    requested_ids = request.trek_ids or recommended_ids
    if not requested_ids:
        requested_ids = list(cards_by_id.keys())
    selected_cards = [cards_by_id[trek_id] for trek_id in requested_ids if trek_id in cards_by_id]
    if not selected_cards:
        raise HTTPException(status_code=400, detail="No requested trek_ids are available in the latest shortlist")

    try:
        query_embedding = query_embedding_service.embed_query(request.question)
        chunks = vector_index.search(
            query_embedding,
            trek_ids=[card.trek_id for card in selected_cards],
            section_types=request.section_types or None,
            limit=request.max_chunks,
        )
    except (EmbeddingProviderError, RetrievalIndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=404, detail="No retrieval chunks found for the selected treks")

    try:
        return rag_answer_service.answer(
            question=request.question,
            onboarding=session.onboarding,
            selected_cards=selected_cards,
            chunks=chunks,
        )
    except RagLlmError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str) -> SessionState:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

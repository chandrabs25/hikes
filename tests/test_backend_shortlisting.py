import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
from fastapi.testclient import TestClient

from backend.app.cards import build_candidate_card
from backend.app.data import TrekRepository
from backend.app.filtering import group_constraints, hard_exclusion_reason, shortlist_treks
from backend.app.llm import RecommendationLlmError, RecommendationService, recommendation_messages, recommendation_schema
from backend.app.rag import _normalise_answer_text, _normalise_suggested_followups, chat_answer_schema, rag_messages
from backend.app.main import app
from backend.app.schemas import (
    Difficulty,
    LlmRecommendation,
    OnboardingState,
    ParticipantProfile,
    RecommendedTrek,
    RetrievalCitation,
    TrekChatResponse,
    TrekComparison,
    TripPreferences,
)


class BackendShortlistingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = TrekRepository()

    def test_loads_all_filter_records(self):
        self.assertEqual(len(self.repo.treks), 34)
        self.assertEqual(len(self.repo.profiles), 34)

    def test_multiple_participants_derive_group_age_and_fitness_constraints(self):
        onboarding = OnboardingState(
            participants=[
                ParticipantProfile(
                    name="Adult",
                    age=41,
                    notes="Comfortable with cold, prefers quieter trails.",
                ),
                ParticipantProfile(
                    name="Child",
                    age=9,
                    notes="First Himalayan trek; gets tired on long climbs.",
                ),
            ]
        )
        constraints = group_constraints(onboarding)
        self.assertEqual(constraints.youngest_age, 9)
        self.assertEqual(constraints.oldest_age, 41)

    def test_target_difficulty_is_selected_band_not_ceiling(self):
        onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(
                target_difficulty=Difficulty.EASY_MODERATE,
            ),
        )
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        self.assertTrue(shortlist.eligible_candidates)
        self.assertTrue(
            all(card.facts.difficulty == "Easy-Moderate" for card in shortlist.eligible_candidates)
        )
        self.assertEqual(len(shortlist.eligible_candidates), 5)

    def test_age_duration_altitude_and_offloading_filters_are_deterministic(self):
        child_onboarding = OnboardingState(participants=[ParticipantProfile(name="Kid", age=8)])
        moderate_trek = next(trek for trek in self.repo.treks if trek.trek_id == "beas-kund")
        self.assertEqual(hard_exclusion_reason(moderate_trek, child_onboarding), "youngest_age_below_trek_minimum")

        duration_onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(duration_days=5),
        )
        dayara = next(trek for trek in self.repo.treks if trek.trek_id == "dayara-bugyal-trek")
        self.assertEqual(hard_exclusion_reason(dayara, duration_onboarding), "duration_mismatch")

        altitude_onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(altitude_ceiling_ft=10_000),
        )
        self.assertEqual(hard_exclusion_reason(dayara, altitude_onboarding), "altitude_exceeds_ceiling")

        offload_onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(needs_offloading=True),
        )
        bali = next(trek for trek in self.repo.treks if trek.trek_id == "bali-pass-ruinsara-tal")
        self.assertEqual(hard_exclusion_reason(bali, offload_onboarding), "offloading_required_but_unavailable")

    def test_shortlist_caps_eligible_candidates_to_six(self):
        onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(target_difficulty=Difficulty.MODERATE),
        )
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        self.assertEqual(len(shortlist.eligible_candidates), 6)
        self.assertTrue(all(card.facts.difficulty == "Moderate" for card in shortlist.eligible_candidates))

    def test_low_match_target_adds_adjacent_conditional_candidates(self):
        onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(target_difficulty=Difficulty.EASY),
        )
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        self.assertEqual(len(shortlist.eligible_candidates), 1)
        self.assertEqual(len(shortlist.conditional_candidates), 2)
        self.assertTrue(
            all(card.facts.difficulty == "Easy-Moderate" for card in shortlist.conditional_candidates)
        )

    def test_candidate_card_excludes_bulky_content(self):
        dayara = next(trek for trek in self.repo.treks if trek.trek_id == "dayara-bugyal-trek")
        card = build_candidate_card(dayara)
        dumped = card.model_dump_json()
        self.assertEqual(card.source_url, "https://indiahikes.com/dayara-bugyal-trek")
        self.assertEqual(
            card.image_url,
            "https://images.prismic.io/indiahike/37356-Dayara-Bugyal-Indiahikes.jpg?auto=format,compress&rect=0,35,797,399&w=1200&h=600",
        )
        self.assertEqual(card.video_url, "https://www.youtube.com/watch?v=MhoyVIeNALM")
        self.assertEqual(card.facts.age_range.min, 8)
        self.assertEqual(card.facts.pickup.city, "Asli Pappu Da Dhaba, Dehradun")
        self.assertEqual(card.facts.dropoff.time, "6.00 PM")
        self.assertTrue(card.facts.offloading)
        self.assertTrue(card.facts.cloakroom)
        self.assertEqual(card.facts.accommodation, "Tents (2-sharing)")
        axes = card.decision_axes
        self.assertLessEqual(len(axes.group_fit.experience), 2)
        self.assertLessEqual(len(axes.group_fit.family_or_child), 2)
        self.assertLessEqual(len(axes.experience_match.snow), 2)
        self.assertLessEqual(len(axes.discomfort_profile.steep_or_strenuous), 2)
        self.assertLessEqual(len(axes.decision_notes.best_for), 3)
        self.assertLessEqual(len(axes.decision_notes.not_ideal_for), 3)
        self.assertLessEqual(len(axes.decision_notes.key_tradeoffs), 3)
        top_level_keys = set(card.model_dump(mode="json").keys())
        self.assertNotIn("fit", top_level_keys)
        self.assertNotIn("experience", top_level_keys)
        self.assertNotIn("watchouts", top_level_keys)
        self.assertNotIn("best_for", top_level_keys)
        self.assertNotIn("avoid_if", top_level_keys)
        self.assertNotIn("faq", dumped.lower())
        self.assertNotIn("itinerary", dumped.lower())
        self.assertNotIn("quote_or_summary", dumped)
        self.assertNotIn("evidence", dumped.lower())

    def test_candidate_card_uses_empty_arrays_for_missing_signals(self):
        dayara = next(trek for trek in self.repo.treks if trek.trek_id == "dayara-bugyal-trek")
        card = build_candidate_card(dayara)
        self.assertEqual(card.decision_axes.experience_match.solitude_or_crowds, [])
        self.assertEqual(card.decision_axes.discomfort_profile.crowds, [])

    def test_llm_prompt_uses_onboarding_and_candidate_cards_only(self):
        onboarding = OnboardingState(
            participants=[
                ParticipantProfile(
                    name="Riya",
                    age=10,
                    notes="First Himalayan trek. Gets tired on long climbs, but loves snow.",
                )
            ],
            preferences=TripPreferences(target_difficulty=Difficulty.EASY_MODERATE),
            text_input=(
                "We want snow, but this should feel confidence-building for a first-time child trekker. "
                "Avoid scary terrain and exhaustion. The child should want to trek again. "
                "Medium cold tolerance, low tolerance for long steep days."
            ),
        )
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        messages = recommendation_messages(
            onboarding,
            shortlist.eligible_candidates,
            shortlist.conditional_candidates,
        )
        payload = messages[1]["content"]
        self.assertIn("group_onboarding", payload)
        self.assertIn("eligible_candidates", payload)
        self.assertIn("decision_axes", payload)
        self.assertIn("text_input", payload)
        self.assertIn("Gets tired on long climbs", payload)
        self.assertIn("confidence-building", payload)
        self.assertIn("Use text_input as the primary source", messages[1]["content"])
        self.assertIn("person_specific_notes", messages[1]["content"])
        self.assertIn("Treat empty candidate fields as unknown", messages[0]["content"])
        self.assertNotIn("source_url", payload)
        self.assertNotIn("faq", payload.lower())
        self.assertNotIn("itinerary", payload.lower())
        self.assertNotIn("quote_or_summary", payload)
        self.assertNotIn("evidence_refs", payload)

    def test_llm_recommendation_schema_allows_four_recommendations(self):
        schema = recommendation_schema()
        self.assertEqual(schema["properties"]["recommended"]["maxItems"], 4)
        self.assertIn(
            "person_specific_notes",
            schema["properties"]["recommended"]["items"]["required"],
        )

    def test_lightweight_participant_notes_are_supported(self):
        onboarding = OnboardingState(
            participants=[
                ParticipantProfile(
                    name="Riya",
                    age=10,
                    notes="First Himalayan trek. Gets tired on long climbs, but loves snow.",
                ),
                ParticipantProfile(
                    name="Arun",
                    age=42,
                    notes="Fine with cold but prefers avoiding very crowded trails.",
                ),
            ],
            preferences=TripPreferences(
                travel_months=["December"],
                target_difficulty=Difficulty.EASY_MODERATE,
                duration_days=6,
            ),
            text_input="We want a confidence-building snow trek, not something scary or exhausting.",
        )
        constraints = group_constraints(onboarding)
        self.assertEqual(constraints.youngest_age, 10)
        messages = recommendation_messages(onboarding, shortlist_treks(self.repo.treks, onboarding).eligible_candidates, [])
        payload = messages[1]["content"]
        self.assertIn("Riya", payload)
        self.assertIn("Gets tired on long climbs", payload)
        self.assertIn("confidence-building snow trek", payload)

    def test_recommendation_service_requires_api_key(self):
        onboarding = OnboardingState(participants=[ParticipantProfile(name="A", age=30)])
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RecommendationLlmError) as ctx:
                RecommendationService().recommend(onboarding, shortlist)
        self.assertIn("FIREWORKS_API_KEY", str(ctx.exception))

    def test_recommendation_service_parses_live_response(self):
        onboarding = OnboardingState(
            participants=[ParticipantProfile(name="A", age=30)],
            preferences=TripPreferences(target_difficulty=Difficulty.EASY_MODERATE),
        )
        shortlist = shortlist_treks(self.repo.treks, onboarding)
        fake = LlmRecommendation(
            mode="live",
            recommended=[
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Best overall fit",
                    reasons=["Beginner-friendly"],
                    tradeoffs=["Winter snow can raise difficulty"],
                    person_specific_notes=["Riya should be watched on long climbs."],
                )
            ],
            comparison=[
                TrekComparison(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    best_fit_for=["First Himalayan trek"],
                    concerns=["Snow if travelling in winter"],
                )
            ],
            questions_to_refine=["Which month are you travelling?"],
            notes="Patched live response",
        )
        with patch.dict("os.environ", {"FIREWORKS_API_KEY": "test-key"}):
            with patch("backend.app.llm.call_fireworks_recommendation", return_value=fake):
                recommendation = RecommendationService().recommend(onboarding, shortlist)
        self.assertEqual(recommendation.mode, "live")
        self.assertEqual(recommendation.recommended[0].trek_id, "dayara-bugyal-trek")

    def test_rag_prompt_uses_group_profile_trek_refs_and_retrieved_chunks(self):
        onboarding = OnboardingState(
            participants=[ParticipantProfile(name="Riya", age=10, notes="First trek.")],
            text_input="We want to compare winter snow options.",
        )
        dayara = next(trek for trek in self.repo.treks if trek.trek_id == "dayara-bugyal-trek")
        card = build_candidate_card(dayara)
        chunks = [
            {
                "chunk_id": "dayara::faq::1",
                "trek_id": "dayara-bugyal-trek",
                "trek_title": "Dayara Bugyal Trek",
                "section_type": "faq",
                "title": "Snow",
                "text": "Dayara can have snow in winter.",
            }
        ]
        messages = rag_messages(
            question="Will there be snow?",
            onboarding=onboarding,
            selected_cards=[card],
            chunks=chunks,
        )
        payload = messages[1]["content"]
        self.assertIn("retrieved_chunks", payload)
        self.assertIn("selected_treks", payload)
        self.assertIn("Dayara can have snow in winter", payload)
        self.assertIn("dayara-bugyal-trek", payload)
        self.assertNotIn("decision_axes", payload)
        self.assertNotIn("source_url", payload)
        self.assertIn("Do not use outside knowledge", messages[0]["content"])
        self.assertIn("retrieved_chunks", messages[0]["content"])
        self.assertEqual(chat_answer_schema()["properties"]["suggested_followups"]["maxItems"], 3)

    def test_rag_normalises_nested_json_answer_text(self):
        data = {
            "answer": '{"answer":"Line one\\\\n\\\\n**Line two**","suggested_followups":["What about snow?"]}',
            "suggested_followups": [],
        }
        self.assertEqual(_normalise_answer_text(data["answer"]), "Line one\n\n**Line two**")
        self.assertEqual(_normalise_suggested_followups(data), ["What about snow?"])


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_filter_options(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["trek_count"], 34)

        options = self.client.get("/treks/filter-options")
        self.assertEqual(options.status_code, 200)
        self.assertIn("Moderate", options.json()["difficulty_buckets"])
        self.assertTrue(options.json()["pickup_cities"])

    def test_session_onboarding_shortlist_round_trip(self):
        created = self.client.post("/sessions", json={"trip_name": "Family trek"})
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]

        onboarding_payload = {
            "participants": [
                {
                    "name": "Parent",
                    "age": 38,
                    "notes": "Has done one Himalayan trek. Fine with cold but prefers quieter trails.",
                },
                {
                    "name": "Child",
                    "age": 10,
                    "notes": "First Himalayan trek. Gets tired on long climbs, but loves snow.",
                },
            ],
            "preferences": {
                "travel_months": ["December"],
                "target_difficulty": "Easy-Moderate",
                "duration_days": 6,
                "themes": ["snow", "meadows"],
                "avoid": ["technical terrain"],
            },
            "text_input": "We want snow, but the child should feel confident, not overwhelmed. Avoid scary terrain. This should be a joyful first Himalayan trek.",
        }
        updated = self.client.put(f"/sessions/{session_id}/onboarding", json=onboarding_payload)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(len(updated.json()["onboarding"]["participants"]), 2)
        self.assertIn("joyful first Himalayan trek", updated.json()["onboarding"]["text_input"])

        fake = LlmRecommendation(
            mode="live",
            recommended=[
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Best fit",
                    reasons=["Beginner-friendly"],
                    tradeoffs=[],
                    person_specific_notes=["Child should be supported on snowy sections."],
                )
            ],
            comparison=[],
            questions_to_refine=[],
            notes="Live reasoning complete",
        )
        with patch("backend.app.main.recommendation_service.recommend", return_value=fake):
            shortlist = self.client.post(f"/sessions/{session_id}/shortlist")
        self.assertEqual(shortlist.status_code, 200)
        body = shortlist.json()
        self.assertLessEqual(len(body["eligible_candidates"]), 6)
        self.assertEqual(body["llm_recommendation"]["mode"], "live")

        fetched = self.client.get(f"/sessions/{session_id}")
        self.assertEqual(fetched.status_code, 200)
        self.assertIsNotNone(fetched.json()["latest_shortlist"])

        table = self.client.get(f"/sessions/{session_id}/comparison-table")
        self.assertEqual(table.status_code, 200)
        table_body = table.json()
        self.assertEqual(table_body["source"], "latest_llm_recommendations")
        self.assertEqual([row["trek_id"] for row in table_body["rows"]], ["dayara-bugyal-trek"])
        row = table_body["rows"][0]
        self.assertEqual(row["difficulty"], "Easy-Moderate")
        self.assertEqual(row["duration_days"], 6)
        self.assertEqual(row["altitude_ft"], 11830)
        self.assertEqual(row["age_range"]["min"], 8)
        self.assertEqual(row["fitness"], "5 km in 40 min")
        self.assertEqual(row["pickup"]["city"], "Asli Pappu Da Dhaba, Dehradun")
        self.assertTrue(row["offloading"])

    def test_minimal_ux_onboarding_payload_round_trip(self):
        created = self.client.post("/sessions", json={"trip_name": "Minimal family trek"})
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["session_id"]
        payload = {
            "participants": [
                {
                    "name": "Riya",
                    "age": 10,
                    "notes": "First Himalayan trek. Gets tired on long climbs, but loves snow.",
                },
                {
                    "name": "Arun",
                    "age": 42,
                    "notes": "Fine with cold but prefers avoiding very crowded trails.",
                },
            ],
            "preferences": {
                "travel_months": ["December"],
                "target_difficulty": "Easy-Moderate",
                "duration_days": 6,
            },
            "text_input": "We want a confidence-building snow trek, not something scary or exhausting.",
        }
        response = self.client.put(f"/sessions/{session_id}/onboarding", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()["onboarding"]
        self.assertEqual(body["participants"][0]["notes"], payload["participants"][0]["notes"])
        self.assertEqual(body["preferences"]["target_difficulty"], "Easy-Moderate")
        self.assertIn("confidence-building", body["text_input"])

    def test_removed_onboarding_fields_are_rejected(self):
        created = self.client.post("/sessions", json={"trip_name": "Old fields"})
        session_id = created.json()["session_id"]
        payload = {
            "participants": [
                {
                    "name": "Riya",
                    "age": 10,
                    "fitness": {"can_run_5k": True, "five_k_time_min": 40},
                }
            ],
            "preferences": {
                "target_difficulty": "Easy-Moderate",
                "max_days": 7,
                "max_difficulty": "Moderate",
            },
        }
        response = self.client.put(f"/sessions/{session_id}/onboarding", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_comparison_table_requires_shortlist(self):
        created = self.client.post("/sessions", json={"trip_name": "No shortlist"})
        session_id = created.json()["session_id"]
        response = self.client.get(f"/sessions/{session_id}/comparison-table")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Run shortlist", response.json()["detail"])

    def test_comparison_table_follows_llm_recommendation_order(self):
        created = self.client.post("/sessions", json={"trip_name": "Order trek"})
        session_id = created.json()["session_id"]
        payload = {
            "participants": [{"name": "Adult", "age": 30}],
            "preferences": {"target_difficulty": "Easy-Moderate"},
        }
        self.client.put(f"/sessions/{session_id}/onboarding", json=payload)
        fake = LlmRecommendation(
            mode="live",
            recommended=[
                RecommendedTrek(
                    trek_id="kedarkantha-trek",
                    title="Kedarkantha Trek",
                    recommendation="Top pick",
                    reasons=[],
                    tradeoffs=[],
                    person_specific_notes=[],
                ),
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Alternative",
                    reasons=[],
                    tradeoffs=[],
                    person_specific_notes=[],
                ),
            ],
            comparison=[],
            questions_to_refine=[],
            notes="Live reasoning complete",
        )
        with patch("backend.app.main.recommendation_service.recommend", return_value=fake):
            self.client.post(f"/sessions/{session_id}/shortlist")

        response = self.client.get(f"/sessions/{session_id}/comparison-table")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [row["trek_id"] for row in response.json()["rows"]],
            ["kedarkantha-trek", "dayara-bugyal-trek"],
        )

    def test_shortlist_endpoint_returns_503_when_llm_unavailable(self):
        created = self.client.post("/sessions", json={"trip_name": "No key trek"})
        session_id = created.json()["session_id"]
        self.client.put(
            f"/sessions/{session_id}/onboarding",
            json={"participants": [{"name": "Adult", "age": 30}], "preferences": {}},
        )
        with patch(
            "backend.app.main.recommendation_service.recommend",
            side_effect=RecommendationLlmError("Set FIREWORKS_API_KEY to enable live LLM reasoning."),
        ):
            response = self.client.post(f"/sessions/{session_id}/shortlist")
        self.assertEqual(response.status_code, 503)
        self.assertIn("FIREWORKS_API_KEY", response.json()["detail"])

    def test_shortlist_endpoint_returns_live_recommendation_shape(self):
        created = self.client.post("/sessions", json={"trip_name": "LLM trek"})
        session_id = created.json()["session_id"]
        payload = {
            "participants": [{"name": "Adult", "age": 30}],
            "preferences": {"target_difficulty": "Easy-Moderate"},
        }
        self.client.put(f"/sessions/{session_id}/onboarding", json=payload)

        fake = LlmRecommendation(
            mode="live",
            recommended=[
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Best fit",
                    reasons=["Easy-Moderate and beginner-friendly"],
                    tradeoffs=[],
                    person_specific_notes=[],
                )
            ],
            comparison=[],
            questions_to_refine=[],
            notes="Live reasoning complete",
        )
        with patch("backend.app.main.recommendation_service.recommend", return_value=fake):
            response = self.client.post(f"/sessions/{session_id}/shortlist")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["llm_recommendation"]["mode"], "live")
        self.assertEqual(body["llm_recommendation"]["recommended"][0]["trek_id"], "dayara-bugyal-trek")

    def test_chat_endpoint_retrieves_only_recommended_treks_by_default(self):
        created = self.client.post("/sessions", json={"trip_name": "Chat trek"})
        session_id = created.json()["session_id"]
        payload = {
            "participants": [{"name": "Adult", "age": 30}],
            "preferences": {"target_difficulty": "Easy-Moderate"},
        }
        self.client.put(f"/sessions/{session_id}/onboarding", json=payload)
        fake_recommendation = LlmRecommendation(
            mode="live",
            recommended=[
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Best fit",
                    reasons=[],
                    tradeoffs=[],
                    person_specific_notes=[],
                )
            ],
            comparison=[],
            questions_to_refine=[],
            notes="Live reasoning complete",
        )
        with patch("backend.app.main.recommendation_service.recommend", return_value=fake_recommendation):
            self.client.post(f"/sessions/{session_id}/shortlist")

        fake_chunks = [
            {
                "chunk_id": "dayara::faq::1",
                "trek_id": "dayara-bugyal-trek",
                "trek_title": "Dayara Bugyal Trek",
                "section_type": "faq",
                "title": "Snow",
                "source_url": "https://indiahikes.com/dayara-bugyal-trek",
                "score": 0.82,
                "text": "Dayara can have snow in winter.",
            }
        ]
        fake_answer = TrekChatResponse(
            answer="Dayara can have snow in winter.",
            suggested_followups=["Do you want a day-wise view?"],
            citations=[
                RetrievalCitation(
                    chunk_id="dayara::faq::1",
                    trek_id="dayara-bugyal-trek",
                    trek_title="Dayara Bugyal Trek",
                    section_type="faq",
                    title="Snow",
                    source_url="https://indiahikes.com/dayara-bugyal-trek",
                    score=0.82,
                )
            ],
            used_trek_ids=["dayara-bugyal-trek"],
        )
        with patch("backend.app.retrieval.TrekVectorIndex.available", new_callable=PropertyMock) as available, patch(
            "backend.app.main.query_embedding_service.embed_query",
            return_value=np.asarray([1.0, 0.0], dtype=np.float32),
        ), patch("backend.app.main.vector_index.search", return_value=fake_chunks) as search, patch(
            "backend.app.main.rag_answer_service.answer",
            return_value=fake_answer,
        ):
            available.return_value = True
            response = self.client.post(
                f"/sessions/{session_id}/chat",
                json={"question": "Will there be snow?"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "live")
        self.assertEqual(body["used_trek_ids"], ["dayara-bugyal-trek"])
        self.assertEqual(body["citations"][0]["chunk_id"], "dayara::faq::1")
        self.assertEqual(search.call_args.kwargs["trek_ids"], ["dayara-bugyal-trek"])

    def test_chat_requires_shortlist_and_vector_index(self):
        created = self.client.post("/sessions", json={"trip_name": "No chat yet"})
        session_id = created.json()["session_id"]
        response = self.client.post(f"/sessions/{session_id}/chat", json={"question": "How hard is it?"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Run shortlist", response.json()["detail"])

        self.client.put(
            f"/sessions/{session_id}/onboarding",
            json={"participants": [{"name": "Adult", "age": 30}], "preferences": {}},
        )
        fake_recommendation = LlmRecommendation(
            recommended=[
                RecommendedTrek(
                    trek_id="dayara-bugyal-trek",
                    title="Dayara Bugyal Trek",
                    recommendation="Best fit",
                    reasons=[],
                    tradeoffs=[],
                    person_specific_notes=[],
                )
            ]
        )
        with patch("backend.app.main.recommendation_service.recommend", return_value=fake_recommendation):
            self.client.post(f"/sessions/{session_id}/shortlist")
        with patch("backend.app.retrieval.TrekVectorIndex.available", new_callable=PropertyMock) as available:
            available.return_value = False
            response = self.client.post(f"/sessions/{session_id}/chat", json={"question": "How hard is it?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("vector_index", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

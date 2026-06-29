from __future__ import annotations

from uuid import uuid4

from backend.app.schemas import OnboardingState, SessionState, ShortlistResponse


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, trip_name: str | None = None) -> SessionState:
        session = SessionState(session_id=str(uuid4()), trip_name=trip_name)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def set_onboarding(self, session_id: str, onboarding: OnboardingState) -> SessionState | None:
        session = self.get(session_id)
        if session is None:
            return None
        session.onboarding = onboarding
        session.latest_shortlist = None
        return session

    def set_shortlist(self, session_id: str, shortlist: ShortlistResponse) -> SessionState | None:
        session = self.get(session_id)
        if session is None:
            return None
        session.latest_shortlist = shortlist
        return session

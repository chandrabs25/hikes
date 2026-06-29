from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORS_ORIGINS = ["http://127.0.0.1:3000", "http://localhost:3000"]


def load_backend_env() -> None:
    load_dotenv(BACKEND_DIR / ".env")


def cors_origins() -> list[str]:
    raw = os.getenv("BACKEND_CORS_ORIGINS")
    if not raw:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]

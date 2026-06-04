"""Shared payload helpers for baseline runs."""

from __future__ import annotations

from pathlib import Path

from .embeddings import load_embeddings_pickle


def load_saved_embedding_payload(path: str | Path) -> dict[str, object]:
    """Load a saved embedding payload and ensure baseline-required fields exist."""

    payload = load_embeddings_pickle(path)
    if not isinstance(payload, dict):
        raise TypeError("Saved embeddings must contain a dictionary payload.")
    missing = {"embeddings", "tokens"} - set(payload)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"Saved embeddings payload is missing: {missing_fields}")

    payload = dict(payload)
    payload.setdefault("saved_path", str(path))
    return payload

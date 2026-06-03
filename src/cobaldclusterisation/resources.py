"""Helpers for locating or cloning external CoBaLD data repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


COBALD_RUS_REPO = "https://github.com/CobaldAnnotation/CobaldRus.git"
SEMANTIC_HIERARCHY_REPO = "https://github.com/CobaldAnnotation/semantic-hierarchy.git"
DEFAULT_HIERARCHY_FILENAME = "hyperonims_hierarchy.csv"


@dataclass(slots=True)
class ExternalDataPaths:
    """Local paths to cloned external data repositories."""

    corpus_dir: Path
    hierarchy_repo_dir: Path
    hierarchy_csv: Path


def _clone_if_missing(repo_url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", repo_url, str(destination)],
        check=True,
    )


def resolve_hierarchy_path(path: str | Path) -> Path:
    """Resolve a hierarchy CSV path from a file or cloned repository directory."""

    path = Path(path)
    if path.is_file():
        return path
    candidate = path / DEFAULT_HIERARCHY_FILENAME
    if candidate.is_file():
        return candidate
    matches = sorted(path.rglob(DEFAULT_HIERARCHY_FILENAME)) if path.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {DEFAULT_HIERARCHY_FILENAME!r} under {path}."
    )


def ensure_external_data(
    *,
    root_dir: str | Path = ".",
    corpus_dir_name: str = "CobaldRus",
    hierarchy_dir_name: str = "semantic-hierarchy",
    corpus_repo: str = COBALD_RUS_REPO,
    hierarchy_repo: str = SEMANTIC_HIERARCHY_REPO,
) -> ExternalDataPaths:
    """Clone the CoBaLD corpus and semantic hierarchy repos if missing."""

    root = Path(root_dir)
    corpus_dir = root / corpus_dir_name
    hierarchy_repo_dir = root / hierarchy_dir_name
    _clone_if_missing(corpus_repo, corpus_dir)
    _clone_if_missing(hierarchy_repo, hierarchy_repo_dir)
    hierarchy_csv = resolve_hierarchy_path(hierarchy_repo_dir)
    return ExternalDataPaths(
        corpus_dir=corpus_dir,
        hierarchy_repo_dir=hierarchy_repo_dir,
        hierarchy_csv=hierarchy_csv,
    )

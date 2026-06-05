"""Semantic-class frequency statistics for CoBaLD datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import corpus_to_dataframe, load_corpus


INVALID_SEMCLASS_VALUES = {"", "_", "nan", "None"}


def semclass_counts(tokens: pd.DataFrame) -> pd.Series:
    """Return descending frequency counts for valid SEMCLASS labels."""

    if "SEMCLASS" not in tokens.columns:
        raise ValueError("tokens must contain a SEMCLASS column")
    semclasses = tokens["SEMCLASS"].astype(str)
    valid = semclasses[~semclasses.isin(INVALID_SEMCLASS_VALUES)]
    return valid.value_counts()


def build_semclass_threshold_table(
    counts: pd.Series,
    *,
    max_occurrences: int = 15,
) -> pd.DataFrame:
    """Build a cumulative table of classes present at least N times."""

    if max_occurrences < 1:
        raise ValueError("max_occurrences must be positive")

    rows: list[dict[str, object]] = []
    for minimum_occurrences in range(1, max_occurrences + 1):
        covered = counts[counts >= minimum_occurrences]
        rows.append(
            {
                "minimum_occurrences": minimum_occurrences,
                "description": (
                    "total semantic classes"
                    if minimum_occurrences == 1
                    else f"semantic classes present at least {minimum_occurrences} times"
                ),
                "semclass_count": int(len(covered)),
                "labeled_token_count": int(covered.sum()),
            }
        )
    return pd.DataFrame(rows)


def semclass_threshold_table_from_corpus(
    data_dir: str | Path = "CobaldRus",
    *,
    splits: Sequence[str] = ("train", "dev"),
    max_occurrences: int = 15,
) -> pd.DataFrame:
    """Load CoBaLD splits and build the SEMCLASS threshold table."""

    sentences = load_corpus(data_dir, splits=splits)
    tokens = corpus_to_dataframe(
        sentences,
        include_punctuation=False,
        include_empty=False,
    )
    return build_semclass_threshold_table(
        semclass_counts(tokens),
        max_occurrences=max_occurrences,
    )


def semclass_table_to_markdown(table: pd.DataFrame) -> str:
    """Format a semantic-class threshold table as Markdown."""

    columns = list(table.columns)
    rows = [
        [str(row[column]) for column in columns]
        for row in table.to_dict(orient="records")
    ]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]

    def format_row(values: Sequence[object]) -> str:
        cells = [
            str(value).ljust(width)
            for value, width in zip(values, widths, strict=True)
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join(
        [
            format_row(columns),
            separator,
            *(format_row(row) for row in rows),
        ]
    )

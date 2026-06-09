"""Collocation tables for CoBaLD corpus inspection."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .data import Sentence, Token, load_corpus


TokenBasis = Literal["lemma", "form"]

DEFAULT_TOKEN_BASES: tuple[TokenBasis, ...] = ("lemma", "form")
DEFAULT_NGRAM_SIZES: tuple[int, ...] = (2, 3)
DEFAULT_SORT_BY = "pmi"
DEFAULT_MIN_FREQ = 3
VALID_SORT_COLUMNS = {
    "frequency",
    "raw_freq",
    "pmi",
    "likelihood_ratio",
    "student_t",
    "chi_sq",
}


@dataclass(frozen=True, slots=True)
class CollocationConfig:
    """Configuration for CoBaLD collocation extraction."""

    data_dir: str | Path = "CobaldRus"
    splits: Sequence[str] = ("train", "dev")
    token_bases: Sequence[TokenBasis] = DEFAULT_TOKEN_BASES
    ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES
    min_freq: int = DEFAULT_MIN_FREQ
    sort_by: str = DEFAULT_SORT_BY
    max_rows: int | None = None
    lowercase: bool = True
    output_path: str | Path = "outputs/collocations/cobald_collocations.xlsx"


def _require_nltk() -> tuple[object, object, object, object]:
    try:
        from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
        from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
    except ImportError as exc:
        raise ImportError(
            "Collocation extraction requires nltk. Install it in Colab with: "
            'pip install -e ".[collocations]" openpyxl'
        ) from exc
    return (
        BigramCollocationFinder,
        TrigramCollocationFinder,
        BigramAssocMeasures,
        TrigramAssocMeasures,
    )


def _validate_config(config: CollocationConfig) -> None:
    if config.min_freq < 1:
        raise ValueError("min_freq must be positive")
    if config.max_rows is not None and config.max_rows < 1:
        raise ValueError("max_rows must be positive when provided")
    if config.sort_by not in VALID_SORT_COLUMNS:
        valid = ", ".join(sorted(VALID_SORT_COLUMNS))
        raise ValueError(f"sort_by must be one of: {valid}")
    unsupported = sorted(set(config.ngram_sizes) - {2, 3})
    if unsupported:
        raise ValueError(f"Only 2-grams and 3-grams are supported, got: {unsupported}")
    unsupported_bases = sorted(set(config.token_bases) - {"lemma", "form"})
    if unsupported_bases:
        raise ValueError(
            f"token_bases must contain only 'lemma' and/or 'form', got: {unsupported_bases}"
        )


def token_text(token: Token, *, basis: TokenBasis, lowercase: bool = True) -> str | None:
    """Return normalized token text for one token and basis."""

    value = token.lemma if basis == "lemma" else token.form
    if value in {"", "_"}:
        return None
    return value.lower() if lowercase else value


def sentence_token_documents(
    sentences: Iterable[Sentence],
    *,
    basis: TokenBasis,
    lowercase: bool = True,
) -> list[list[str]]:
    """Return sentence-bounded token documents for NLTK collocation finders."""

    documents: list[list[str]] = []
    for sentence in sentences:
        document = [
            text
            for token in sentence.surface_tokens(include_punctuation=False)
            if (text := token_text(token, basis=basis, lowercase=lowercase)) is not None
        ]
        if document:
            documents.append(document)
    return documents


def _score_ngram(
    finder: object,
    ngram: tuple[str, ...],
    measures: object,
    metric_names: Sequence[str],
) -> dict[str, float | int | str]:
    score_ngram: Callable[..., float] = getattr(finder, "score_ngram")
    row: dict[str, float | int | str] = {
        "ngram": " ".join(ngram),
        "frequency": int(finder.ngram_fd[ngram]),
    }
    for metric_name in metric_names:
        metric = getattr(measures, metric_name)
        row[metric_name] = float(score_ngram(metric, *ngram))
    return row


def collocation_table_from_documents(
    documents: Sequence[Sequence[str]],
    *,
    ngram_size: int,
    min_freq: int = DEFAULT_MIN_FREQ,
    sort_by: str = DEFAULT_SORT_BY,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Build a scored collocation table from sentence-bounded token documents."""

    if ngram_size not in {2, 3}:
        raise ValueError("ngram_size must be 2 or 3")
    if min_freq < 1:
        raise ValueError("min_freq must be positive")
    if sort_by not in VALID_SORT_COLUMNS:
        valid = ", ".join(sorted(VALID_SORT_COLUMNS))
        raise ValueError(f"sort_by must be one of: {valid}")

    (
        BigramCollocationFinder,
        TrigramCollocationFinder,
        BigramAssocMeasures,
        TrigramAssocMeasures,
    ) = _require_nltk()

    if ngram_size == 2:
        finder = BigramCollocationFinder.from_documents(documents)
        measures = BigramAssocMeasures()
    else:
        finder = TrigramCollocationFinder.from_documents(documents)
        measures = TrigramAssocMeasures()
    finder.apply_freq_filter(min_freq)

    metric_names = ("raw_freq", "pmi", "likelihood_ratio", "student_t", "chi_sq")
    rows = [
        _score_ngram(finder, ngram, measures, metric_names)
        for ngram in finder.ngram_fd
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "ngram",
                "frequency",
                "raw_freq",
                "pmi",
                "likelihood_ratio",
                "student_t",
                "chi_sq",
            ]
        )

    table = table.sort_values(
        by=[sort_by, "frequency", "ngram"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if max_rows is not None:
        table = table.head(max_rows).reset_index(drop=True)
    return table


def build_collocation_tables(
    sentences: Sequence[Sentence],
    *,
    token_bases: Sequence[TokenBasis] = DEFAULT_TOKEN_BASES,
    ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
    min_freq: int = DEFAULT_MIN_FREQ,
    sort_by: str = DEFAULT_SORT_BY,
    max_rows: int | None = None,
    lowercase: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build all requested collocation tables for loaded CoBaLD sentences."""

    config = CollocationConfig(
        token_bases=token_bases,
        ngram_sizes=ngram_sizes,
        min_freq=min_freq,
        sort_by=sort_by,
        max_rows=max_rows,
        lowercase=lowercase,
    )
    _validate_config(config)

    tables: dict[str, pd.DataFrame] = {}
    for basis in token_bases:
        documents = sentence_token_documents(
            sentences,
            basis=basis,
            lowercase=lowercase,
        )
        for ngram_size in ngram_sizes:
            name = f"{basis}_{'bigrams' if ngram_size == 2 else 'trigrams'}"
            table = collocation_table_from_documents(
                documents,
                ngram_size=ngram_size,
                min_freq=min_freq,
                sort_by=sort_by,
                max_rows=max_rows,
            )
            table.insert(0, "token_basis", basis)
            table.insert(1, "ngram_size", ngram_size)
            tables[name] = table
    return tables


def _metadata_table(
    *,
    config: CollocationConfig,
    sentences: Sequence[Sentence],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    token_counts = {
        basis: sum(
            len(document)
            for document in sentence_token_documents(
                sentences,
                basis=basis,
                lowercase=config.lowercase,
            )
        )
        for basis in config.token_bases
    }
    rows: list[dict[str, object]] = [
        {"key": "data_dir", "value": str(config.data_dir)},
        {"key": "splits", "value": ",".join(config.splits)},
        {"key": "sentence_count", "value": len(sentences)},
        {"key": "token_bases", "value": ",".join(config.token_bases)},
        {"key": "ngram_sizes", "value": ",".join(str(n) for n in config.ngram_sizes)},
        {"key": "min_freq", "value": config.min_freq},
        {"key": "sort_by", "value": config.sort_by},
        {"key": "max_rows", "value": "" if config.max_rows is None else config.max_rows},
        {"key": "lowercase", "value": config.lowercase},
        {"key": "token_filter", "value": "surface non-punctuation tokens"},
    ]
    rows.extend(
        {"key": f"{basis}_token_count", "value": count}
        for basis, count in token_counts.items()
    )
    rows.extend(
        {"key": f"{name}_rows", "value": len(table)}
        for name, table in tables.items()
    )
    return pd.DataFrame(rows)


def write_collocation_excel(
    tables: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    metadata: pd.DataFrame | None = None,
) -> str:
    """Write collocation tables to an Excel workbook."""

    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.sheets[sheet_name[:31]]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
        if metadata is not None:
            metadata.to_excel(writer, sheet_name="metadata", index=False)
            worksheet = writer.sheets["metadata"]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
    return str(output_path)


def run_collocation_experiment(
    config: CollocationConfig | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Load CoBaLD data, score collocations, and write an Excel workbook."""

    if config is None:
        config = CollocationConfig(**kwargs)
    elif kwargs:
        raise ValueError("Pass either config or keyword arguments, not both")
    _validate_config(config)

    sentences = load_corpus(config.data_dir, splits=config.splits)
    tables = build_collocation_tables(
        sentences,
        token_bases=config.token_bases,
        ngram_sizes=config.ngram_sizes,
        min_freq=config.min_freq,
        sort_by=config.sort_by,
        max_rows=config.max_rows,
        lowercase=config.lowercase,
    )
    metadata = _metadata_table(config=config, sentences=sentences, tables=tables)
    output_path = write_collocation_excel(
        tables,
        config.output_path,
        metadata=metadata,
    )
    return {
        "config": config,
        "sentences": sentences,
        "tables": tables,
        "metadata": metadata,
        "output_path": output_path,
    }

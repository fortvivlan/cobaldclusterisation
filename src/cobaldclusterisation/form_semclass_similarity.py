"""Analyze identical token forms with divergent CoBaLD semantic labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import pandas as pd

from .embeddings import load_embeddings_pickle


INVALID_SEMCLASS_VALUES = {"", "_", "nan", "None"}
DEFAULT_ARTIFACT_PATH = (
    "drive/MyDrive/cobald_outputs/results/"
    "100,200,300,400,512,565cl_rubert_tiny2_Minibatch_Kmeans_artifacts.pkl"
)
DEFAULT_OUTPUT_NAME = "identical_form_semclass_similarity.xlsx"
DEFAULT_RUBERT_EMBEDDINGS_NAME = "rubert_tiny2_cobald.pkl"
IDENTITY_COLUMNS = (
    "FORM",
    "LEMMA",
    "SEMCLASS",
    "split",
    "sentence_index",
    "token_index",
)


@dataclass(frozen=True, slots=True)
class FormSemclassSimilarityConfig:
    """Configuration for the identical-form SEMCLASS similarity experiment."""

    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH
    embeddings_path: str | Path | None = None
    output_path: str | Path | None = None
    max_examples_per_run: int = 200
    examples_per_form: int = 4
    create_plots: bool = True


def load_clustering_artifact(path: str | Path) -> dict[str, object]:
    """Load a clustering ``*_artifacts.pkl`` file."""

    with Path(path).open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("runs"), list):
        raise ValueError("artifact must be a dictionary containing a 'runs' list")
    return artifact


def infer_embeddings_path(
    artifact_path: str | Path,
    artifact: dict[str, object] | None = None,
) -> Path:
    """Infer the aligned embedding payload path for a clustering artifact."""

    artifact_path = Path(artifact_path)
    candidates: list[Path] = []
    metadata = artifact.get("metadata", {}) if artifact is not None else {}
    if isinstance(metadata, dict):
        for key in ("embeddings", "embeddings_path", "embedding_path"):
            value = metadata.get(key)
            if value:
                candidates.append(Path(str(value)))
                candidates.append(artifact_path.parent / str(value))

    drive_root = (
        artifact_path.parent.parent
        if artifact_path.parent.name == "results"
        else artifact_path.parent
    )
    candidates.extend(
        [
            drive_root / DEFAULT_RUBERT_EMBEDDINGS_NAME,
            drive_root / "rubert_tiny2.pkl",
            artifact_path.parent / DEFAULT_RUBERT_EMBEDDINGS_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not infer embeddings_path for the clustering artifact. "
        "Pass embeddings_path explicitly. Checked:\n"
        f"{checked}"
    )


def _valid_semclass_mask(values: pd.Series) -> pd.Series:
    semclasses = values.astype(str)
    return ~semclasses.isin(INVALID_SEMCLASS_VALUES)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 0,
    )
    return normalized.astype(np.float32, copy=False)


def _series_for_compare(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").fillna("<NA>")


def validate_alignment(
    embedding_tokens: pd.DataFrame,
    embeddings: np.ndarray,
    runs: Sequence[dict[str, object]],
) -> None:
    """Validate that embeddings, payload tokens, and artifact token rows align."""

    if len(embedding_tokens) != len(embeddings):
        raise ValueError(
            f"Embedding token rows ({len(embedding_tokens)}) do not match "
            f"embedding rows ({len(embeddings)})."
        )
    if not runs:
        raise ValueError("artifact contains no runs")

    reference = embedding_tokens.reset_index(drop=True)
    for run_index, run in enumerate(runs):
        annotated = run.get("annotated_tokens")
        if not isinstance(annotated, pd.DataFrame):
            raise ValueError(f"run {run_index} has no annotated_tokens DataFrame")
        annotated = annotated.reset_index(drop=True)
        if len(annotated) != len(reference):
            raise ValueError(
                f"run {run_index} annotated token rows ({len(annotated)}) do not "
                f"match embedding token rows ({len(reference)})."
            )
        for column in IDENTITY_COLUMNS:
            if column not in reference.columns or column not in annotated.columns:
                continue
            mismatches = _series_for_compare(reference, column) != _series_for_compare(
                annotated,
                column,
            )
            if bool(mismatches.any()):
                mismatch_count = int(mismatches.sum())
                raise ValueError(
                    f"run {run_index} annotated_tokens mismatch embedding tokens in "
                    f"column {column!r}: {mismatch_count} rows differ."
                )


def _pair_count(count: int) -> int:
    return count * (count - 1) // 2


def _within_sum(vector_sum: np.ndarray, count: int) -> float:
    if count < 2:
        return 0.0
    return float(np.dot(vector_sum, vector_sum) - count) / 2.0


def _mean_or_nan(total: float, count: int) -> float:
    return float(total / count) if count else float("nan")


def _format_counts(values: pd.Series, *, max_items: int = 12) -> str:
    counts = values.astype(str).value_counts().head(max_items)
    return "\n".join(f"{key}: {int(value)}" for key, value in counts.items())


def _format_counter(counter: Counter[int], *, max_items: int = 12) -> str:
    return "\n".join(
        f"{key}: {value}"
        for key, value in counter.most_common(max_items)
    )


def _shared_cluster_count(a: Counter[int], b: Counter[int]) -> int:
    return sum(a[cluster] * b[cluster] for cluster in set(a) & set(b))


def _cluster_pair_count(counter: Counter[int]) -> int:
    return sum(_pair_count(count) for count in counter.values())


def _group_stats(
    form_lower: str,
    group: pd.DataFrame,
    vectors: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    semclass_parts: dict[str, dict[str, object]] = {}
    for semclass, part in group.groupby("SEMCLASS", sort=True):
        indices = part["_row_index"].to_numpy(dtype=np.int64)
        vector_sum = vectors[indices].sum(axis=0)
        clusters = Counter(int(value) for value in part["cluster"])
        count = int(len(part))
        pair_count = _pair_count(count)
        same_cluster_pairs = _cluster_pair_count(clusters)
        semclass_parts[str(semclass)] = {
            "count": count,
            "indices": indices,
            "vector_sum": vector_sum,
            "clusters": clusters,
            "pair_count": pair_count,
            "cosine_sum": _within_sum(vector_sum, count),
            "same_cluster_pairs": same_cluster_pairs,
        }

    same_pair_count = 0
    same_cosine_sum = 0.0
    same_different_cluster_pairs = 0
    within_rows: list[dict[str, object]] = []
    for semclass, stats in semclass_parts.items():
        pair_count = int(stats["pair_count"])
        same_cluster_pairs = int(stats["same_cluster_pairs"])
        different_cluster_pairs = pair_count - same_cluster_pairs
        same_pair_count += pair_count
        same_cosine_sum += float(stats["cosine_sum"])
        same_different_cluster_pairs += different_cluster_pairs
        within_rows.append(
            {
                "form_lower": form_lower,
                "semclass": semclass,
                "token_count": int(stats["count"]),
                "pair_count": pair_count,
                "mean_cosine": _mean_or_nan(float(stats["cosine_sum"]), pair_count),
                "same_cluster_pair_count": same_cluster_pairs,
                "different_cluster_pair_count": different_cluster_pairs,
                "different_cluster_rate": _mean_or_nan(
                    float(different_cluster_pairs),
                    pair_count,
                ),
                "cluster_count": len(stats["clusters"]),
                "clusters": _format_counter(stats["clusters"]),
            }
        )

    classpair_rows: list[dict[str, object]] = []
    diff_pair_count = 0
    diff_cosine_sum = 0.0
    diff_same_cluster_pairs = 0
    semclasses = sorted(semclass_parts)
    for left_index, semclass_a in enumerate(semclasses):
        stats_a = semclass_parts[semclass_a]
        for semclass_b in semclasses[left_index + 1 :]:
            stats_b = semclass_parts[semclass_b]
            pair_count = int(stats_a["count"]) * int(stats_b["count"])
            cosine_sum = float(
                np.dot(
                    np.asarray(stats_a["vector_sum"]),
                    np.asarray(stats_b["vector_sum"]),
                )
            )
            same_cluster_pairs = _shared_cluster_count(
                stats_a["clusters"],
                stats_b["clusters"],
            )
            diff_pair_count += pair_count
            diff_cosine_sum += cosine_sum
            diff_same_cluster_pairs += same_cluster_pairs
            classpair_rows.append(
                {
                    "form_lower": form_lower,
                    "semclass_a": semclass_a,
                    "semclass_b": semclass_b,
                    "token_count_a": int(stats_a["count"]),
                    "token_count_b": int(stats_b["count"]),
                    "pair_count": pair_count,
                    "mean_cosine": _mean_or_nan(cosine_sum, pair_count),
                    "same_cluster_pair_count": same_cluster_pairs,
                    "same_cluster_rate": _mean_or_nan(
                        float(same_cluster_pairs),
                        pair_count,
                    ),
                    "clusters_a": _format_counter(stats_a["clusters"]),
                    "clusters_b": _format_counter(stats_b["clusters"]),
                    "shared_clusters": "\n".join(
                        str(cluster)
                        for cluster in sorted(
                            set(stats_a["clusters"]) & set(stats_b["clusters"])
                        )
                    ),
                }
            )

    same_mean = _mean_or_nan(same_cosine_sum, same_pair_count)
    diff_mean = _mean_or_nan(diff_cosine_sum, diff_pair_count)
    form_row = {
        "form_lower": form_lower,
        "token_count": int(len(group)),
        "semclass_count": int(group["SEMCLASS"].nunique()),
        "cluster_count": int(group["cluster"].nunique()),
        "forms": _format_counts(group["FORM"]),
        "semclasses": _format_counts(group["SEMCLASS"]),
        "clusters": _format_counts(group["cluster"]),
        "diff_semclass_pair_count": diff_pair_count,
        "diff_semclass_mean_cosine": diff_mean,
        "diff_semclass_same_cluster_pair_count": diff_same_cluster_pairs,
        "diff_semclass_same_cluster_rate": _mean_or_nan(
            float(diff_same_cluster_pairs),
            diff_pair_count,
        ),
        "same_semclass_pair_count": same_pair_count,
        "same_semclass_mean_cosine": same_mean,
        "same_semclass_different_cluster_pair_count": same_different_cluster_pairs,
        "same_semclass_different_cluster_rate": _mean_or_nan(
            float(same_different_cluster_pairs),
            same_pair_count,
        ),
        "same_minus_diff_cosine": (
            same_mean - diff_mean
            if not np.isnan(same_mean) and not np.isnan(diff_mean)
            else float("nan")
        ),
    }
    return form_row, classpair_rows, within_rows


def _run_label(run_index: int, run: dict[str, object]) -> str:
    config = run.get("config", {})
    if not isinstance(config, dict):
        return f"run{run_index}"
    algorithm = str(config.get("algorithm", "run"))
    if "n_clusters" in config and config.get("n_clusters") is not None:
        return f"k{int(config['n_clusters'])}"
    if algorithm in {"knn_leiden", "knn_louvain"}:
        return f"nn{config.get('graph_n_neighbors', run_index)}"
    if algorithm == "hdbscan":
        return f"min{config.get('min_cluster_size', run_index)}"
    return f"run{run_index}"


def _sheet_name(base: str, used: set[str]) -> str:
    candidate = base[:31]
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while True:
        suffix = f"_{index}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _annotated_for_run(
    embedding_tokens: pd.DataFrame,
    run: dict[str, object],
    run_index: int,
) -> pd.DataFrame:
    annotated = run.get("annotated_tokens")
    if not isinstance(annotated, pd.DataFrame):
        raise ValueError(f"run {run_index} has no annotated_tokens DataFrame")
    frame = annotated.reset_index(drop=True).copy()
    for column in ("embedding_index", "context_text", "embedding_context_text"):
        if column not in frame.columns and column in embedding_tokens:
            frame[column] = embedding_tokens[column].to_numpy()
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["form_lower"] = frame["FORM"].astype(str).str.casefold()
    frame = frame.loc[_valid_semclass_mask(frame["SEMCLASS"])].copy()
    ambiguous_forms = (
        frame.groupby("form_lower")["SEMCLASS"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    return frame.loc[frame["form_lower"].isin(ambiguous_forms)].copy()


def analyze_run(
    embedding_tokens: pd.DataFrame,
    embeddings: np.ndarray,
    run: dict[str, object],
    *,
    run_index: int,
    max_examples: int,
    examples_per_form: int,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Analyze one clustering run from an artifact."""

    vectors = _normalize_embeddings(embeddings)
    frame = _annotated_for_run(embedding_tokens, run, run_index)
    form_rows: list[dict[str, object]] = []
    classpair_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    for form_lower, group in frame.groupby("form_lower", sort=True):
        form_row, group_classpair_rows, group_within_rows = _group_stats(
            str(form_lower),
            group,
            vectors,
        )
        form_rows.append(form_row)
        classpair_rows.extend(group_classpair_rows)
        within_rows.extend(group_within_rows)

    forms = pd.DataFrame(form_rows)
    classpairs = pd.DataFrame(classpair_rows)
    within = pd.DataFrame(within_rows)
    if not forms.empty:
        forms = forms.sort_values(
            by=[
                "same_minus_diff_cosine",
                "diff_semclass_same_cluster_rate",
                "same_semclass_different_cluster_rate",
                "diff_semclass_pair_count",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    if not classpairs.empty:
        classpairs = classpairs.sort_values(
            by=["mean_cosine", "same_cluster_rate", "pair_count"],
            ascending=[True, False, False],
            na_position="last",
        ).reset_index(drop=True)
    if not within.empty:
        within = within.sort_values(
            by=["different_cluster_rate", "pair_count"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

    examples = _example_rows(
        frame,
        forms,
        max_examples=max_examples,
        examples_per_form=examples_per_form,
    )
    overview = _overview_row(run, run_index, forms)
    return {
        "overview": overview,
        "forms": forms,
        "classpairs": classpairs,
        "within": within,
        "examples": examples,
    }


def _overview_row(
    run: dict[str, object],
    run_index: int,
    forms: pd.DataFrame,
) -> dict[str, object]:
    config = run.get("config", {})
    config = config if isinstance(config, dict) else {}
    if forms.empty:
        totals = {
            "diff_pairs": 0,
            "same_pairs": 0,
            "diff_same_cluster_pairs": 0,
            "same_different_cluster_pairs": 0,
        }
        diff_mean = same_mean = float("nan")
    else:
        diff_pairs = int(forms["diff_semclass_pair_count"].sum())
        same_pairs = int(forms["same_semclass_pair_count"].sum())
        diff_mean = _mean_or_nan(
            float(
                (
                    forms["diff_semclass_mean_cosine"]
                    * forms["diff_semclass_pair_count"]
                ).sum()
            ),
            diff_pairs,
        )
        same_mean = _mean_or_nan(
            float(
                (
                    forms["same_semclass_mean_cosine"]
                    * forms["same_semclass_pair_count"]
                ).sum()
            ),
            same_pairs,
        )
        totals = {
            "diff_pairs": diff_pairs,
            "same_pairs": same_pairs,
            "diff_same_cluster_pairs": int(
                forms["diff_semclass_same_cluster_pair_count"].sum()
            ),
            "same_different_cluster_pairs": int(
                forms["same_semclass_different_cluster_pair_count"].sum()
            ),
        }
    return {
        "run_index": run_index,
        "run_label": _run_label(run_index, run),
        "algorithm": config.get("algorithm"),
        "n_clusters": config.get("n_clusters"),
        "ambiguous_form_count": int(len(forms)),
        "ambiguous_token_count": int(forms["token_count"].sum()) if not forms.empty else 0,
        "diff_semclass_pair_count": totals["diff_pairs"],
        "diff_semclass_mean_cosine": diff_mean,
        "same_semclass_pair_count": totals["same_pairs"],
        "same_semclass_mean_cosine": same_mean,
        "same_minus_diff_cosine": (
            same_mean - diff_mean
            if not np.isnan(same_mean) and not np.isnan(diff_mean)
            else float("nan")
        ),
        "diff_semclass_same_cluster_rate": _mean_or_nan(
            float(totals["diff_same_cluster_pairs"]),
            totals["diff_pairs"],
        ),
        "same_semclass_different_cluster_rate": _mean_or_nan(
            float(totals["same_different_cluster_pairs"]),
            totals["same_pairs"],
        ),
    }


def _example_rows(
    frame: pd.DataFrame,
    forms: pd.DataFrame,
    *,
    max_examples: int,
    examples_per_form: int,
) -> pd.DataFrame:
    if frame.empty or forms.empty or max_examples <= 0 or examples_per_form <= 0:
        return pd.DataFrame()
    selected_forms = forms["form_lower"].head(
        max(1, (max_examples + examples_per_form - 1) // examples_per_form)
    )
    rows: list[dict[str, object]] = []
    for form_lower in selected_forms:
        form_rows = frame.loc[frame["form_lower"] == form_lower]
        for _, row in (
            form_rows.sort_values(["SEMCLASS", "cluster"])
            .groupby("SEMCLASS", sort=True, group_keys=False)
            .head(examples_per_form)
            .iterrows()
        ):
            rows.append(
                {
                    "form_lower": form_lower,
                    "FORM": row.get("FORM"),
                    "LEMMA": row.get("LEMMA"),
                    "SEMCLASS": row.get("SEMCLASS"),
                    "cluster": row.get("cluster"),
                    "AUTO_SEMCLASS": row.get("AUTO_SEMCLASS"),
                    "split": row.get("split"),
                    "sentence_index": row.get("sentence_index"),
                    "token_index": row.get("token_index"),
                    "sentence_id": row.get("sentence_id"),
                    "context_text": row.get("context_text"),
                    "embedding_context_text": row.get("embedding_context_text"),
                    "embedding_index": row.get("embedding_index"),
                }
            )
            if len(rows) >= max_examples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def write_similarity_excel(
    run_tables: Sequence[dict[str, pd.DataFrame | dict[str, object]]],
    output_path: str | Path,
) -> str:
    """Write all run-level similarity tables to an Excel workbook."""

    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame([tables["overview"] for tables in run_tables])
    used_sheets = {"overview"}
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="overview", index=False)
        for tables in run_tables:
            overview_row = tables["overview"]
            assert isinstance(overview_row, dict)
            run_label = str(overview_row["run_label"])
            for table_name in ("forms", "classpairs", "within", "examples"):
                table = tables[table_name]
                assert isinstance(table, pd.DataFrame)
                sheet_name = _sheet_name(f"{table_name}_{run_label}", used_sheets)
                table.to_excel(writer, sheet_name=sheet_name, index=False)
        for worksheet in writer.sheets.values():
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
    return str(output_path)


def write_similarity_plots(
    run_tables: Sequence[dict[str, pd.DataFrame | dict[str, object]]],
    output_path: str | Path,
) -> list[str]:
    """Write optional diagnostic scatter plots if matplotlib is installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_path = Path(output_path)
    plot_paths: list[str] = []
    for tables in run_tables:
        overview = tables["overview"]
        forms = tables["forms"]
        if not isinstance(overview, dict) or not isinstance(forms, pd.DataFrame):
            continue
        if forms.empty:
            continue
        run_label = str(overview["run_label"])
        scatter_path = output_path.with_name(
            f"{output_path.stem}_{run_label}_cosine_scatter.png"
        )
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            forms["diff_semclass_mean_cosine"],
            forms["same_semclass_mean_cosine"],
            alpha=0.35,
            s=12,
        )
        axis.set_xlabel("Different SEMCLASS mean cosine")
        axis.set_ylabel("Same SEMCLASS mean cosine")
        axis.set_title(f"Identical-form cosine similarity: {run_label}")
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(scatter_path))

        cluster_path = output_path.with_name(
            f"{output_path.stem}_{run_label}_cluster_scatter.png"
        )
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(
            forms["same_minus_diff_cosine"],
            forms["diff_semclass_same_cluster_rate"],
            alpha=0.35,
            s=12,
        )
        axis.set_xlabel("Same minus different SEMCLASS cosine")
        axis.set_ylabel("Different SEMCLASS same-cluster rate")
        axis.set_title(f"Cosine gap vs cluster merging: {run_label}")
        fig.tight_layout()
        fig.savefig(cluster_path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(cluster_path))
    return plot_paths


def run_form_semclass_similarity_experiment(
    config: FormSemclassSimilarityConfig | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Run the identical-form SEMCLASS similarity experiment."""

    if config is None:
        config = FormSemclassSimilarityConfig(**kwargs)
    elif kwargs:
        raise ValueError("Pass either config or keyword arguments, not both")
    if config.max_examples_per_run < 0:
        raise ValueError("max_examples_per_run must be non-negative")
    if config.examples_per_form < 1:
        raise ValueError("examples_per_form must be positive")

    artifact_path = Path(config.artifact_path)
    artifact = load_clustering_artifact(artifact_path)
    embeddings_path = (
        Path(config.embeddings_path)
        if config.embeddings_path is not None
        else infer_embeddings_path(artifact_path, artifact)
    )
    embedding_payload = load_embeddings_pickle(embeddings_path)
    embeddings = np.asarray(embedding_payload["embeddings"], dtype=np.float32)
    embedding_tokens = embedding_payload["tokens"]
    if not isinstance(embedding_tokens, pd.DataFrame):
        raise ValueError("embedding payload 'tokens' must be a DataFrame")
    runs = artifact["runs"]
    validate_alignment(embedding_tokens, embeddings, runs)

    run_tables = [
        analyze_run(
            embedding_tokens,
            embeddings,
            run,
            run_index=run_index,
            max_examples=config.max_examples_per_run,
            examples_per_form=config.examples_per_form,
        )
        for run_index, run in enumerate(runs)
    ]
    output_path = (
        Path(config.output_path)
        if config.output_path is not None
        else artifact_path.with_name(DEFAULT_OUTPUT_NAME)
    )
    excel_path = write_similarity_excel(run_tables, output_path)
    plot_paths = (
        write_similarity_plots(run_tables, output_path)
        if config.create_plots
        else []
    )
    return {
        "artifact": artifact,
        "artifact_path": str(artifact_path),
        "embeddings_path": str(embeddings_path),
        "run_tables": run_tables,
        "output_path": excel_path,
        "plot_paths": plot_paths,
    }

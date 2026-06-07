"""Shared exports for clustering result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .data import CONLLU_COLUMNS
from .scores import _run_part, write_scores
from .summary_excel import write_cluster_summary_excel, write_hierarchy_alignment_excel


HIERARCHICAL_ALGORITHMS = {"bisecting_kmeans", "birch", "knn_leiden"}
FLAT_ALGORITHMS = {"kmeans", "minibatch_kmeans"}
CONLLU_PLUS_AUTO_COLUMN = "AUTO_SEMCLASS"


@dataclass(slots=True)
class ClusteringExportPaths:
    """Drive-facing clustering result paths."""

    excel: str
    hierarchy_alignment_excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]
    artifacts_pickle: str
    annotated_conllu_plus: str | None
    annotated_conllu_plus_files: list[str]


def default_results_dir(
    output_dir: str | Path,
    drive_dir: str | Path,
    *,
    default_output_dir: str | Path,
) -> Path:
    """Use Drive results for default Colab output, otherwise keep local output."""

    output_path = Path(output_dir)
    if output_path == Path(default_output_dir):
        return Path(drive_dir) / "results"
    return output_path


def output_prefix(
    results: Sequence[dict[str, object]],
    *,
    label: str,
    family: str | None = None,
) -> str:
    """Build a stable, non-overwriting filename prefix for a result suite."""

    counts = _cluster_count_part(results)
    resolved_family = family or _infer_family(results)
    return _safe_file_component(f"{counts}cl_{label}_{resolved_family}")


def write_clustering_outputs(
    results: Sequence[dict[str, object]],
    tokens: pd.DataFrame,
    *,
    output_dir: str | Path,
    label: str,
    family: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ClusteringExportPaths:
    """Write reusable, human-readable, and score artifacts for clustering runs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix(results, label=label, family=family)

    excel_path = write_cluster_summary_excel(
        results,
        output_path / f"{prefix}_summary.xlsx",
    )
    hierarchy_alignment_path = write_hierarchy_alignment_excel(
        results,
        output_path / f"{prefix}_hierarchy_alignment.xlsx",
    )
    scores_csv, scores_xlsx, scores_txt = write_scores(
        results,
        output_dir=output_path,
        label=prefix,
    )

    runs: list[dict[str, object]] = []
    conllu_paths: list[str] = []
    for index, result in enumerate(results):
        labels = np.asarray(result["labels"], dtype=np.int64)
        annotated = annotate_tokens(tokens, labels, result=result)
        conllu_path = output_path / (
            f"{prefix}_{index}_{_result_algorithm(result)}_"
            f"{_run_part(result.get('config', {}))}_annotated.conllu"
        )
        write_conllu_plus(annotated, conllu_path)
        conllu_paths.append(str(conllu_path))
        runs.append(
            {
                "config": result.get("config", {}),
                "labels": labels,
                "annotated_tokens": annotated,
                "summary": result.get("summary"),
                "scores": result.get("scores", {}),
                "hierarchy_alignment": result.get("hierarchy_alignment"),
            }
        )

    artifact_path = output_path / f"{prefix}_artifacts.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(
            {
                "format_version": 1,
                "label": label,
                "prefix": prefix,
                "metadata": metadata or {},
                "runs": runs,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return ClusteringExportPaths(
        excel=excel_path,
        hierarchy_alignment_excel=hierarchy_alignment_path,
        scores_csv=scores_csv,
        scores_xlsx=scores_xlsx,
        scores_txt=scores_txt,
        artifacts_pickle=str(artifact_path),
        annotated_conllu_plus=conllu_paths[0] if conllu_paths else None,
        annotated_conllu_plus_files=conllu_paths,
    )


def annotate_tokens(
    tokens: pd.DataFrame,
    labels: Sequence[int],
    *,
    result: dict[str, object],
) -> pd.DataFrame:
    """Attach automatic cluster labels to a CoBaLD token table."""

    labels_array = np.asarray(labels, dtype=np.int64)
    if len(tokens) != len(labels_array):
        raise ValueError("tokens and labels must have the same length")
    annotated = tokens.reset_index(drop=True).copy()
    label_names = _cluster_label_names(result)
    auto_labels = [
        label_names.get(int(label), f"cluster_{int(label)}")
        for label in labels_array
    ]
    annotated.insert(0, "cluster", labels_array)
    annotated[CONLLU_PLUS_AUTO_COLUMN] = auto_labels
    return annotated


def write_conllu_plus(tokens: pd.DataFrame, output_path: str | Path) -> str:
    """Write target-token CoNLL-U Plus with automatic semantic class labels."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# global.columns = "
        + " ".join((*CONLLU_COLUMNS, CONLLU_PLUS_AUTO_COLUMN))
    ]

    group_columns = [
        column
        for column in ("split", "sentence_index", "sentence_id")
        if column in tokens.columns
    ]
    if group_columns:
        grouped = tokens.groupby(group_columns, sort=False, dropna=False)
        row_groups = [group for _, group in grouped]
    else:
        row_groups = [tokens]

    for group in row_groups:
        first = group.iloc[0] if not group.empty else {}
        sentence_id = _clean_optional(first.get("sentence_id"))
        split = _clean_optional(first.get("split"))
        if sentence_id:
            lines.append(f"# sent_id = {sentence_id}")
        if split:
            lines.append(f"# split = {split}")
        for fallback_id, (_, row) in enumerate(group.iterrows(), start=1):
            columns = [
                _clean_conllu_value(
                    row.get(column, str(fallback_id) if column == "ID" else "_")
                )
                for column in CONLLU_COLUMNS
            ]
            columns.append(_clean_conllu_value(row.get(CONLLU_PLUS_AUTO_COLUMN, "_")))
            lines.append("\t".join(columns))
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(output_path)


def _cluster_label_names(result: dict[str, object]) -> dict[int, str]:
    summary = result.get("summary")
    if not isinstance(summary, pd.DataFrame) or "cluster" not in summary.columns:
        return {}
    names: dict[int, str] = {}
    for _, row in summary.iterrows():
        cluster = int(row["cluster"])
        cluster_name = _clean_auto_label(row.get("cluster_name", ""))
        names[cluster] = (
            f"cluster_{cluster}:{cluster_name}" if cluster_name else f"cluster_{cluster}"
        )
    return names


def _cluster_count_part(results: Sequence[dict[str, object]]) -> str:
    counts: list[int] = []
    for result in results:
        config = result.get("config", {})
        algorithm = str(config.get("algorithm", ""))
        if algorithm in {"hdbscan", "knn_leiden", "knn_louvain"}:
            continue
        try:
            count = int(config.get("n_clusters"))
        except (TypeError, ValueError):
            continue
        if count not in counts:
            counts.append(count)
    if not counts:
        return "auto"
    return ",".join(str(count) for count in counts)


def _infer_family(results: Sequence[dict[str, object]]) -> str:
    algorithms = {_result_algorithm(result) for result in results}
    if algorithms and algorithms <= HIERARCHICAL_ALGORITHMS:
        return "Hierarchical"
    if algorithms == {"minibatch_kmeans"}:
        return "Minibatch_Kmeans"
    if algorithms and algorithms <= FLAT_ALGORITHMS:
        return "Flat"
    return "Mixed"


def _result_algorithm(result: dict[str, object]) -> str:
    return str(result.get("config", {}).get("algorithm", "run"))


def _safe_file_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_,.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def _clean_conllu_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "_"
    text = str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return text if text else "_"


def _clean_auto_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _clean_optional(value: object) -> str:
    text = _clean_conllu_value(value)
    return "" if text in {"", "_", "nan"} else text

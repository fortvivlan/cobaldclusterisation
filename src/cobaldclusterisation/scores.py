"""Score formatting and export helpers for baseline clustering runs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


def metric_reference(metric_name: str) -> str:
    """Return the short score guide used in the README examples."""

    if metric_name == "silhouette":
        return "-1 to 1; higher is better"
    if metric_name == "calinski_harabasz":
        return "0 to +inf; higher is better"
    if metric_name == "davies_bouldin":
        return "0 to +inf; lower is better"
    if metric_name == "n_clusters_found":
        return "count; compare with requested or discovered cluster count"
    if metric_name == "n_noise":
        return "count; HDBSCAN noise points, lower is usually better"
    if metric_name.endswith("_adjusted_rand"):
        return "-1 to 1; higher is better, 0 is near random"
    if metric_name.endswith(
        (
            "_normalized_mutual_info",
            "_homogeneity",
            "_completeness",
            "_v_measure",
            "_purity",
        )
    ):
        return "0 to 1; higher is better"
    if metric_name.endswith("_n_labeled"):
        return "count of tokens with usable SEMCLASS labels"
    return "inspect manually"


def format_score_value(metric_name: str, value: object) -> str:
    """Format score values consistently for printing and text files."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return "nan"
        if metric_name in {"n_clusters_found", "n_noise"} or metric_name.endswith(
            "_n_labeled"
        ):
            return str(int(value))
        return f"{float(value):.4f}"
    return str(value)


def format_score_table(title: str, scores: dict[str, object]) -> str:
    """Build a printable score table."""

    lines = [title, "metric - result - reference note"]
    for metric_name, value in scores.items():
        lines.append(
            f"{metric_name} - "
            f"{format_score_value(metric_name, value)} - "
            f"{metric_reference(metric_name)}"
        )
    return "\n".join(lines)


def print_result_scores(label: str, result: dict[str, object]) -> str:
    """Print and return one formatted result score table."""

    config = result["config"]
    algorithm = str(config["algorithm"])
    if algorithm == "hdbscan":
        run_details = (
            f"min_cluster_size={config.get('min_cluster_size')}, "
            f"min_samples={config.get('min_samples')}"
        )
    else:
        run_details = f"k={config.get('n_clusters')}"
    run_name = (
        f"{label}: {algorithm}, "
        f"{run_details}, "
        f"normalize={config.get('normalize')}"
    )
    table = format_score_table(run_name, result["scores"])
    print(table)
    return table


def scores_to_metric_dataframe(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Convert baseline run scores to a metric-row comparison table."""

    metric_names: list[str] = []
    for result in results:
        for metric_name in result.get("scores", {}):
            metric_name = str(metric_name)
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        row: dict[str, object] = {
            "metric": metric_name,
            "reference_note": metric_reference(metric_name),
        }
        for index, result in enumerate(results):
            row[_run_column_name(index, result)] = result.get("scores", {}).get(
                metric_name,
                np.nan,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def _run_column_name(index: int, result: dict[str, object]) -> str:
    config = result.get("config", {})
    algorithm = str(config.get("algorithm", "run"))
    if algorithm == "hdbscan":
        run_part = f"min{config.get('min_cluster_size')}"
    else:
        run_part = f"k{config.get('n_clusters')}"
    return f"run_{index + 1}_{algorithm}_{run_part}"


def _run_file_part(index: int, result: dict[str, object]) -> str:
    config = result.get("config", {})
    algorithm = str(config.get("algorithm", "run"))
    run_part = (
        f"min{config.get('min_cluster_size')}"
        if algorithm == "hdbscan"
        else f"k{config.get('n_clusters')}"
    )
    return f"{index}_{algorithm}_{run_part}"


def _write_score_excel(scores: pd.DataFrame, output_path: Path) -> None:
    try:
        from openpyxl.styles import Alignment
    except ImportError as exc:
        raise ImportError(
            "Excel export requires openpyxl. Install it in Colab with: "
            "pip install openpyxl"
        ) from exc

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="scores", index=False)
        worksheet = writer.sheets["scores"]
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        worksheet.freeze_panes = "A2"


def write_scores(
    results: Sequence[dict[str, object]],
    *,
    output_dir: Path,
    label: str,
) -> tuple[str, str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_filename(label)
    scores = scores_to_metric_dataframe(results)

    scores_csv = output_dir / f"{safe_label}_scores.csv"
    scores.to_csv(scores_csv, index=False)
    scores_xlsx = output_dir / f"{safe_label}_scores.xlsx"
    _write_score_excel(scores, scores_xlsx)

    score_txt_paths: list[str] = []
    for index, result in enumerate(results):
        table = print_result_scores(f"{label} run {index + 1}", result)
        scores_txt = output_dir / (
            f"{safe_label}_{_run_file_part(index, result)}_scores.txt"
        )
        scores_txt.write_text(table + "\n", encoding="utf-8")
        score_txt_paths.append(str(scores_txt))
        print()

    return str(scores_csv), str(scores_xlsx), score_txt_paths


_write_scores = write_scores

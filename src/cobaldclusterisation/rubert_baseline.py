"""Colab-friendly ruBERT-tiny2 baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .clustering import ClusterConfig, run_clustering, scores_to_dataframe
from .embeddings import EmbeddingConfig, generate_embeddings_from_corpus


MODEL_NAME = "cointegrated/rubert-tiny2"
DEFAULT_COLAB_DIR = Path("/content")
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/cobald_outputs")


@dataclass(slots=True)
class RubertBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    excel: str
    scores_csv: str
    scores_txt: list[str]


def _canonical_algorithm_name(name: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", name.strip().lower())
    aliases = {
        "kmeans": "kmeans",
        "k_means": "kmeans",
        "minibatchkmeans": "minibatch_kmeans",
        "mini_batch_kmeans": "minibatch_kmeans",
        "minibatch_kmeans": "minibatch_kmeans",
        "hdbscan": "hdbscan",
        "agglomerative": "agglomerative",
        "agglomerativeclustering": "agglomerative",
        "agglomerative_clustering": "agglomerative",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unknown algorithm. Use any of: KMeans, MiniBatchKMeans, "
            "Agglomerative, HDBSCAN."
        ) from exc


def _as_list(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def build_cluster_configs(
    *,
    algorithms: Sequence[str] = ("KMeans", "MiniBatchKMeans"),
    n_clusters: int | Sequence[int] = 100,
    random_state: int = 42,
    normalize: bool = True,
    batch_size: int = 4096,
    n_init: int = 10,
    min_samples: int = 10,
    min_cluster_size: int = 10,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = None,
    metric: str = "euclidean",
) -> list[ClusterConfig]:
    """Create clustering configs from Colab-friendly algorithm names."""

    cluster_counts = _as_list(n_clusters)
    configs: list[ClusterConfig] = []
    for algorithm_name in algorithms:
        algorithm = _canonical_algorithm_name(algorithm_name)
        if algorithm == "hdbscan":
            configs.append(
                ClusterConfig(
                    algorithm=algorithm,
                    random_state=random_state,
                    normalize=normalize,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                    cluster_selection_method=hdbscan_cluster_selection_method,
                    allow_single_cluster=hdbscan_allow_single_cluster,
                    n_jobs=hdbscan_n_jobs,
                    metric=metric,
                )
            )
            continue

        for cluster_count in cluster_counts:
            configs.append(
                ClusterConfig(
                    algorithm=algorithm,
                    n_clusters=cluster_count,
                    random_state=random_state,
                    normalize=normalize,
                    batch_size=batch_size,
                    n_init=n_init,
                    min_samples=min_samples,
                    metric=metric,
                )
            )
    return configs


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


def write_cluster_summary_excel(
    results: Sequence[dict[str, object]],
    output_path: str | Path,
) -> str:
    """Write all cluster summaries to one Excel workbook."""

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
        for index, result in enumerate(results):
            config = result["config"]
            algorithm = str(config["algorithm"])
            cluster_part = (
                f"min{config.get('min_cluster_size')}"
                if algorithm == "hdbscan"
                else f"k{config.get('n_clusters')}"
            )
            sheet_name = f"{index}_{algorithm}_{cluster_part}"[:31]
            result["summary"].to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            worksheet.freeze_panes = "A2"
    return str(output_path)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def _write_scores(
    results: Sequence[dict[str, object]],
    *,
    output_dir: Path,
    label: str,
) -> tuple[str, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / f"{_safe_filename(label)}_scores.csv"
    scores_to_dataframe(results).to_csv(scores_csv, index=False)

    score_txt_paths: list[str] = []
    for index, result in enumerate(results):
        table = print_result_scores(f"{label} run {index + 1}", result)
        config = result["config"]
        algorithm = str(config["algorithm"])
        run_part = (
            f"min{config.get('min_cluster_size')}"
            if algorithm == "hdbscan"
            else f"k{config.get('n_clusters')}"
        )
        scores_txt = output_dir / (
            f"{_safe_filename(label)}_{index}_{algorithm}_{run_part}_scores.txt"
        )
        scores_txt.write_text(table + "\n", encoding="utf-8")
        score_txt_paths.append(str(scores_txt))
        print()

    return str(scores_csv), score_txt_paths


def run(
    *,
    data_dir: str | Path = "CobaldRus",
    hierarchy: str | Path | pd.DataFrame | None = "hyperonims_hierarchy.csv",
    splits: Sequence[str] = ("train", "dev"),
    algorithms: Sequence[str] = ("KMeans", "MiniBatchKMeans"),
    n_clusters: int | Sequence[int] = 100,
    output_dir: str | Path = DEFAULT_COLAB_DIR,
    save_embeddings_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    embedding_filename: str = "rubert_tiny2_cobald.pkl",
    excel_filename: str = "rubert_tiny2_cluster_summaries.xlsx",
    label: str = "rubert_tiny2",
    embedding_batch_size: int = 32,
    clustering_batch_size: int = 4096,
    max_length: int = 512,
    device: str | None = "cuda",
    seed: int = 42,
    include_punctuation_context: bool = True,
    normalize_embeddings: bool = False,
    normalize_for_clustering: bool = True,
    n_init: int = 10,
    min_samples: int = 10,
    min_cluster_size: int = 10,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = None,
    metric: str = "euclidean",
    hierarchy_depths: Sequence[int] = (1, 2, 3),
    show_progress: bool = True,
) -> dict[str, object]:
    """Run the full ruBERT-tiny2 embedding and clustering baseline.

    The function expects the CoBaLD corpus files, hierarchy CSV, and optional
    Google Drive mount to already exist in the Colab runtime.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_path = output_dir / embedding_filename
    drive_output_dir = Path(drive_dir) if save_embeddings_to_drive else None

    embedding_config = EmbeddingConfig(
        model_name=MODEL_NAME,
        batch_size=embedding_batch_size,
        device=device,
        max_length=max_length,
        seed=seed,
        normalize=normalize_embeddings,
        include_punctuation_context=include_punctuation_context,
    )

    payload = generate_embeddings_from_corpus(
        data_dir=data_dir,
        output_path=embedding_output_path,
        drive_dir=drive_output_dir,
        splits=splits,
        config=embedding_config,
        show_progress=show_progress,
    )
    print(f"embeddings_shape={payload['embeddings'].shape}")
    print(f"tokens={len(payload['tokens'])}")
    print(f"embeddings_saved={payload['saved_path']}")

    configs = build_cluster_configs(
        algorithms=algorithms,
        n_clusters=n_clusters,
        random_state=seed,
        normalize=normalize_for_clustering,
        batch_size=clustering_batch_size,
        n_init=n_init,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
        hdbscan_allow_single_cluster=hdbscan_allow_single_cluster,
        hdbscan_n_jobs=hdbscan_n_jobs,
        metric=metric,
    )

    results: list[dict[str, object]] = []
    progress = tqdm(configs, desc="Clustering full dataset", disable=not show_progress)
    for config in progress:
        progress.set_postfix(algorithm=config.algorithm)
        results.append(
            run_clustering(
                payload["embeddings"],
                payload["tokens"],
                config=config,
                hierarchy=hierarchy,
                hierarchy_depths=hierarchy_depths,
                show_progress=show_progress,
            )
        )

    excel_path = write_cluster_summary_excel(results, output_dir / excel_filename)
    scores_csv, scores_txt = _write_scores(results, output_dir=output_dir, label=label)

    paths = RubertBaselinePaths(
        embeddings=str(payload["saved_path"]),
        excel=excel_path,
        scores_csv=scores_csv,
        scores_txt=scores_txt,
    )
    return {
        "embedding_payload": payload,
        "results": results,
        "paths": asdict(paths),
    }

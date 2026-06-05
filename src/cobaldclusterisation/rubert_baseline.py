"""Colab-friendly ruBERT-tiny2 baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Sequence

import pandas as pd
from tqdm.auto import tqdm

from .baseline_payloads import load_saved_embedding_payload
from .clustering import ClusterConfig, infer_semclass_cluster_count, run_clustering
from .embeddings import EmbeddingConfig, generate_embeddings_from_corpus
from .scores import (
    format_score_table,
    format_score_value,
    metric_reference,
    print_result_scores,
    write_scores as _write_scores,
)
from .summary_excel import write_cluster_summary_excel, write_hierarchy_alignment_excel


MODEL_NAME = "cointegrated/rubert-tiny2"
DEFAULT_COLAB_DIR = Path("/content")
DEFAULT_DRIVE_DIR = Path("/content/drive/MyDrive/cobald_outputs")


@dataclass(slots=True)
class RubertBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    excel: str
    hierarchy_alignment_excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]


def _canonical_algorithm_name(name: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", name.strip().lower())
    aliases = {
        "kmeans": "kmeans",
        "k_means": "kmeans",
        "minibatchkmeans": "minibatch_kmeans",
        "mini_batch_kmeans": "minibatch_kmeans",
        "minibatch_kmeans": "minibatch_kmeans",
        "bisectingkmeans": "bisecting_kmeans",
        "bisecting_kmeans": "bisecting_kmeans",
        "hdbscan": "hdbscan",
        "birch": "birch",
        "knnleiden": "knn_leiden",
        "knn_leiden": "knn_leiden",
        "knn_graph_leiden": "knn_leiden",
        "leiden": "knn_leiden",
        "knnlouvain": "knn_louvain",
        "knn_louvain": "knn_louvain",
        "knn_graph_louvain": "knn_louvain",
        "louvain": "knn_louvain",
        "agglomerative": "agglomerative",
        "agglomerativeclustering": "agglomerative",
        "agglomerative_clustering": "agglomerative",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Unknown algorithm. Use any of: KMeans, MiniBatchKMeans, "
            "BisectingKMeans, BIRCH, KNNLeiden, KNNLouvain, "
            "Agglomerative, HDBSCAN."
        ) from exc


def _as_list(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _cluster_counts(
    value: int | str | Sequence[int],
    *,
    tokens: pd.DataFrame | None,
) -> list[int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "data_semclass":
            if tokens is None:
                raise ValueError("n_clusters='data_semclass' requires a token table")
            return [infer_semclass_cluster_count(tokens)]
        return [int(value)]
    return _as_list(value)


def build_cluster_configs(
    *,
    algorithms: Sequence[str] = ("BisectingKMeans", "BIRCH", "KNNLeiden"),
    n_clusters: int | str | Sequence[int] = 565,
    tokens: pd.DataFrame | None = None,
    random_state: int = 42,
    normalize: bool = True,
    batch_size: int = 4096,
    n_init: int = 1,
    min_samples: int = 10,
    min_cluster_size: int = 10,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = None,
    metric: str = "euclidean",
    birch_threshold: float = 0.75,
    birch_branching_factor: int = 100,
    bisecting_strategy: str = "biggest_inertia",
    graph_n_neighbors: int = 15,
    graph_resolution: float = 1.0,
    graph_metric: str = "cosine",
    graph_n_jobs: int | None = -1,
) -> list[ClusterConfig]:
    """Create clustering configs from Colab-friendly algorithm names."""

    cluster_counts = _cluster_counts(n_clusters, tokens=tokens)
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
        if algorithm in {"knn_leiden", "knn_louvain"}:
            configs.append(
                ClusterConfig(
                    algorithm=algorithm,
                    random_state=random_state,
                    normalize=normalize,
                    metric=graph_metric,
                    n_jobs=graph_n_jobs,
                    graph_n_neighbors=graph_n_neighbors,
                    graph_resolution=graph_resolution,
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
                    birch_threshold=birch_threshold,
                    birch_branching_factor=birch_branching_factor,
                    bisecting_strategy=bisecting_strategy,
                )
            )
    return configs


def run(
    *,
    data_dir: str | Path = "CobaldRus",
    hierarchy: str | Path | pd.DataFrame | None = "hyperonims_hierarchy.csv",
    splits: Sequence[str] = ("train", "dev"),
    algorithms: Sequence[str] = ("BisectingKMeans", "BIRCH", "KNNLeiden"),
    n_clusters: int | str | Sequence[int] = "data_semclass",
    output_dir: str | Path = DEFAULT_COLAB_DIR,
    save_embeddings_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    embedding_filename: str = "rubert_tiny2_cobald.pkl",
    embeddings_path: str | Path | None = None,
    excel_filename: str = "rubert_tiny2_cluster_summaries.xlsx",
    hierarchy_alignment_filename: str = "rubert_tiny2_hierarchy_alignment.xlsx",
    label: str = "rubert_tiny2",
    embedding_batch_size: int = 32,
    clustering_batch_size: int = 4096,
    max_length: int = 512,
    device: str | None = "cuda",
    seed: int = 42,
    include_punctuation_context: bool = True,
    normalize_embeddings: bool = False,
    normalize_for_clustering: bool = True,
    n_init: int = 1,
    min_samples: int = 10,
    min_cluster_size: int = 10,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = None,
    metric: str = "euclidean",
    birch_threshold: float = 0.75,
    birch_branching_factor: int = 100,
    bisecting_strategy: str = "biggest_inertia",
    graph_n_neighbors: int = 15,
    graph_resolution: float = 1.0,
    graph_metric: str = "cosine",
    graph_n_jobs: int | None = -1,
    hierarchy_depths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
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

    if embeddings_path is None:
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
    else:
        payload = load_saved_embedding_payload(embeddings_path)
    print(f"embeddings_shape={payload['embeddings'].shape}")
    print(f"tokens={len(payload['tokens'])}")
    print(f"embeddings_saved={payload['saved_path']}")

    configs = build_cluster_configs(
        algorithms=algorithms,
        n_clusters=n_clusters,
        tokens=payload["tokens"],
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
        birch_threshold=birch_threshold,
        birch_branching_factor=birch_branching_factor,
        bisecting_strategy=bisecting_strategy,
        graph_n_neighbors=graph_n_neighbors,
        graph_resolution=graph_resolution,
        graph_metric=graph_metric,
        graph_n_jobs=graph_n_jobs,
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
    hierarchy_alignment_path = write_hierarchy_alignment_excel(
        results,
        output_dir / hierarchy_alignment_filename,
    )
    scores_csv, scores_xlsx, scores_txt = _write_scores(
        results,
        output_dir=output_dir,
        label=label,
    )

    paths = RubertBaselinePaths(
        embeddings=str(payload["saved_path"]),
        excel=excel_path,
        hierarchy_alignment_excel=hierarchy_alignment_path,
        scores_csv=scores_csv,
        scores_xlsx=scores_xlsx,
        scores_txt=scores_txt,
    )
    return {
        "embedding_payload": payload,
        "results": results,
        "paths": asdict(paths),
    }

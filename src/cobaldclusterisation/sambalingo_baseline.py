"""Colab-friendly SambaLingo full-data baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gc
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize as l2_normalize
from tqdm.auto import tqdm

from .clustering import ClusterConfig, run_clustering
from .embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    save_embeddings_pickle,
)
from .rubert_baseline import (
    DEFAULT_COLAB_DIR,
    DEFAULT_DRIVE_DIR,
    _write_scores,
    build_cluster_configs,
    write_cluster_summary_excel,
)


MODEL_NAME = "sambanovasystems/SambaLingo-Russian-Base"


@dataclass(slots=True)
class SambaLingoBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    clustering_features: str
    excel: str
    scores_csv: str
    scores_txt: list[str]


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _batch_ranges(
    total: int,
    batch_size: int,
    *,
    min_final_size: int = 1,
) -> list[range]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ranges = [
        range(start, min(start + batch_size, total))
        for start in range(0, total, batch_size)
    ]
    if len(ranges) > 1 and len(ranges[-1]) < min_final_size:
        ranges[-2] = range(ranges[-2].start, ranges[-1].stop)
        ranges.pop()
    return ranges


def reduce_embeddings_incremental_pca(
    embeddings: np.ndarray,
    *,
    n_components: int = 256,
    batch_size: int = 8192,
    normalize_input: bool = True,
    normalize_output: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Reduce a full embedding matrix with IncrementalPCA for clustering."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if n_components > min(matrix.shape):
        raise ValueError(
            "n_components must be <= min(n_samples, n_features), got "
            f"{n_components} for shape {matrix.shape}"
        )
    if batch_size < n_components:
        raise ValueError(
            "batch_size must be >= n_components for IncrementalPCA, got "
            f"batch_size={batch_size}, n_components={n_components}"
        )

    working = l2_normalize(matrix) if normalize_input else matrix
    reducer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ranges = _batch_ranges(len(working), batch_size, min_final_size=n_components)

    fit_progress = tqdm(ranges, desc="Fitting IncrementalPCA", disable=not show_progress)
    for batch_range in fit_progress:
        reducer.partial_fit(working[batch_range])

    reduced = np.empty((len(working), n_components), dtype=np.float32)
    transform_progress = tqdm(
        ranges,
        desc="Projecting embeddings",
        disable=not show_progress,
    )
    for batch_range in transform_progress:
        reduced[batch_range] = reducer.transform(working[batch_range]).astype(
            np.float32,
            copy=False,
        )

    if normalize_output:
        reduced = l2_normalize(reduced).astype(np.float32, copy=False)
    return reduced


def make_clustering_payload(
    embedding_payload: dict[str, object],
    *,
    projection_n_components: int = 256,
    projection_batch_size: int = 8192,
    normalize_projection_input: bool = True,
    normalize_projection_output: bool = True,
    show_progress: bool = True,
) -> dict[str, object]:
    """Create the reduced embedding payload used for full-data clustering."""

    reduced = reduce_embeddings_incremental_pca(
        np.asarray(embedding_payload["embeddings"], dtype=np.float32),
        n_components=projection_n_components,
        batch_size=projection_batch_size,
        normalize_input=normalize_projection_input,
        normalize_output=normalize_projection_output,
        show_progress=show_progress,
    )
    return {
        "embeddings": reduced,
        "tokens": embedding_payload["tokens"],
        "config": {
            "source_embedding_config": embedding_payload.get("config", {}),
            "source_embedding_path": embedding_payload.get("saved_path"),
            "projection": "incremental_pca",
            "projection_n_components": projection_n_components,
            "projection_batch_size": projection_batch_size,
            "normalize_projection_input": normalize_projection_input,
            "normalize_projection_output": normalize_projection_output,
            "target_policy": "surface_non_punctuation_tokens",
        },
    }


def _guard_quadratic_algorithms(
    configs: Sequence[ClusterConfig],
    *,
    n_rows: int,
    agglomerative_max_rows: int,
    allow_quadratic_algorithms: bool,
) -> None:
    if allow_quadratic_algorithms:
        return
    requested = [
        config.algorithm
        for config in configs
        if config.algorithm == "agglomerative"
    ]
    if requested and n_rows > agglomerative_max_rows:
        raise ValueError(
            "Agglomerative clustering can require quadratic memory/time. "
            f"The full matrix has {n_rows} rows, which exceeds "
            f"agglomerative_max_rows={agglomerative_max_rows}. Set "
            "allow_quadratic_algorithms=True to run it anyway."
        )


def run(
    *,
    data_dir: str | Path = "CobaldRus",
    hierarchy: str | Path | pd.DataFrame | None = "hyperonims_hierarchy.csv",
    splits: Sequence[str] = ("train", "dev"),
    algorithms: Sequence[str] = ("KMeans", "MiniBatchKMeans", "HDBSCAN"),
    n_clusters: int | Sequence[int] = 100,
    output_dir: str | Path = DEFAULT_COLAB_DIR,
    save_embeddings_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    embedding_filename: str = "sambalingo_russian_base_cobald.pkl",
    clustering_features_filename: str | None = None,
    excel_filename: str = "sambalingo_russian_base_cluster_summaries.xlsx",
    label: str = "sambalingo_russian_base",
    embedding_batch_size: int = 1,
    clustering_batch_size: int = 4096,
    max_length: int = 512,
    device: str | None = "cuda",
    torch_dtype: str | None = "float16",
    seed: int = 42,
    include_punctuation_context: bool = True,
    normalize_embeddings: bool = False,
    normalize_for_clustering: bool = False,
    projection_n_components: int = 256,
    projection_batch_size: int = 8192,
    normalize_projection_input: bool = True,
    normalize_projection_output: bool = True,
    n_init: int = 5,
    min_samples: int = 10,
    min_cluster_size: int = 25,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = -1,
    metric: str = "euclidean",
    hierarchy_depths: Sequence[int] = (1, 2, 3),
    agglomerative_max_rows: int = 50_000,
    allow_quadratic_algorithms: bool = False,
    release_cuda_after_embeddings: bool = True,
    show_progress: bool = True,
) -> dict[str, object]:
    """Run the full SambaLingo embedding and clustering baseline.

    The function expects the CoBaLD corpus files, hierarchy CSV, Hugging Face
    authentication if needed, and optional Google Drive mount to already exist
    in the Colab runtime.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_path = output_dir / embedding_filename
    if clustering_features_filename is None:
        clustering_features_filename = (
            "sambalingo_russian_base_cobald_"
            f"pca{projection_n_components}.pkl"
        )
    drive_output_dir = Path(drive_dir) if save_embeddings_to_drive else None

    embedding_config = EmbeddingConfig(
        model_name=MODEL_NAME,
        batch_size=embedding_batch_size,
        device=device,
        torch_dtype=torch_dtype,
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

    if release_cuda_after_embeddings:
        _release_cuda_memory()

    clustering_payload = make_clustering_payload(
        payload,
        projection_n_components=projection_n_components,
        projection_batch_size=projection_batch_size,
        normalize_projection_input=normalize_projection_input,
        normalize_projection_output=normalize_projection_output,
        show_progress=show_progress,
    )
    clustering_features_path = save_embeddings_pickle(
        clustering_payload,
        output_dir / clustering_features_filename,
    )
    clustering_payload["saved_path"] = str(clustering_features_path)
    print(f"clustering_features_shape={clustering_payload['embeddings'].shape}")
    print(f"clustering_features_saved={clustering_features_path}")

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
    _guard_quadratic_algorithms(
        configs,
        n_rows=len(clustering_payload["tokens"]),
        agglomerative_max_rows=agglomerative_max_rows,
        allow_quadratic_algorithms=allow_quadratic_algorithms,
    )

    results: list[dict[str, object]] = []
    progress = tqdm(configs, desc="Clustering full dataset", disable=not show_progress)
    for config in progress:
        progress.set_postfix(algorithm=config.algorithm)
        results.append(
            run_clustering(
                clustering_payload["embeddings"],
                clustering_payload["tokens"],
                config=config,
                hierarchy=hierarchy,
                hierarchy_depths=hierarchy_depths,
                show_progress=show_progress,
            )
        )

    excel_path = write_cluster_summary_excel(results, output_dir / excel_filename)
    scores_csv, scores_txt = _write_scores(results, output_dir=output_dir, label=label)

    paths = SambaLingoBaselinePaths(
        embeddings=str(payload["saved_path"]),
        clustering_features=str(clustering_features_path),
        excel=excel_path,
        scores_csv=scores_csv,
        scores_txt=scores_txt,
    )
    return {
        "embedding_payload": payload,
        "clustering_payload": clustering_payload,
        "results": results,
        "paths": asdict(paths),
    }

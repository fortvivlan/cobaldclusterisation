"""Colab-friendly GigaChat3 full-data baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm.auto import tqdm

from .baseline_payloads import load_saved_embedding_payload
from .clustering import run_clustering
from .embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    save_embeddings_pickle,
)
from .rubert_baseline import (
    DEFAULT_COLAB_DIR,
    DEFAULT_DRIVE_DIR,
    build_cluster_configs,
)
from .sambalingo_baseline import (
    _guard_quadratic_algorithms,
    _release_cuda_memory,
    make_clustering_payload,
)
from .scores import write_scores as _write_scores
from .summary_excel import write_cluster_summary_excel


MODEL_NAME = "ai-sage/GigaChat3-10B-A1.8B-base"


@dataclass(slots=True)
class GigaChatBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    clustering_features: str
    excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]


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
    embedding_filename: str = "gigachat3_10b_a1_8b_base_cobald.pkl",
    embeddings_path: str | Path | None = None,
    clustering_features_filename: str | None = None,
    excel_filename: str = "gigachat3_10b_a1_8b_base_cluster_summaries.xlsx",
    label: str = "gigachat3_10b_a1_8b_base",
    embedding_batch_size: int = 1,
    clustering_batch_size: int = 4096,
    max_length: int = 512,
    device: str | None = "cuda",
    torch_dtype: str | None = "bfloat16",
    seed: int = 42,
    include_punctuation_context: bool = True,
    normalize_embeddings: bool = False,
    normalize_for_clustering: bool = False,
    trust_remote_code: bool = False,
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
    """Run the full GigaChat3 embedding and clustering baseline.

    The function expects the CoBaLD corpus files, hierarchy CSV, optional
    Hugging Face authentication, and optional Google Drive mount to already
    exist in the Colab runtime.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_path = output_dir / embedding_filename
    if clustering_features_filename is None:
        clustering_features_filename = (
            "gigachat3_10b_a1_8b_base_cobald_"
            f"pca{projection_n_components}.pkl"
        )
    drive_output_dir = Path(drive_dir) if save_embeddings_to_drive else None

    if embeddings_path is None:
        embedding_config = EmbeddingConfig(
            model_name=MODEL_NAME,
            batch_size=embedding_batch_size,
            device=device,
            torch_dtype=torch_dtype,
            max_length=max_length,
            seed=seed,
            normalize=normalize_embeddings,
            trust_remote_code=trust_remote_code,
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

    if embeddings_path is None and release_cuda_after_embeddings:
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
    scores_csv, scores_xlsx, scores_txt = _write_scores(
        results,
        output_dir=output_dir,
        label=label,
    )

    paths = GigaChatBaselinePaths(
        embeddings=str(payload["saved_path"]),
        clustering_features=str(clustering_features_path),
        excel=excel_path,
        scores_csv=scores_csv,
        scores_xlsx=scores_xlsx,
        scores_txt=scores_txt,
    )
    return {
        "embedding_payload": payload,
        "clustering_payload": clustering_payload,
        "results": results,
        "paths": asdict(paths),
    }

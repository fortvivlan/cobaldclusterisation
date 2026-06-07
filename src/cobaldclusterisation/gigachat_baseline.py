"""Colab-friendly GigaChat3 full-data baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm.auto import tqdm

from .baseline_payloads import load_saved_embedding_payload
from .cluster_exports import default_results_dir, write_clustering_outputs
from .clustering import run_clustering, run_clustering_with_training_data
from .embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    save_embeddings_pickle,
)
from .rubert_baseline import (
    COBALD_TRAINING_SOURCE,
    DEFAULT_COLAB_DIR,
    DEFAULT_DRIVE_DIR,
    DEFAULT_SYNTAGRUS_DIR,
    DEFAULT_SYNTAGRUS_FILES,
    DEFAULT_SYNTAGRUS_REPO_URL,
    build_cluster_configs,
    prepare_training_and_cobald_payloads,
)
from .sambalingo_baseline import (
    _guard_quadratic_algorithms,
    _release_cuda_memory,
    make_clustering_payload,
    make_training_and_cobald_clustering_payloads,
)
from .scores import write_scores as _write_scores
from .summary_excel import write_cluster_summary_excel, write_hierarchy_alignment_excel


MODEL_NAME = "ai-sage/GigaChat3-10B-A1.8B-base"


@dataclass(slots=True)
class GigaChatBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    clustering_features: str
    training_embeddings: str | None
    training_clustering_features: str | None
    excel: str
    semclass_cluster_map_excel: str
    hierarchy_alignment_excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]
    artifacts_pickle: str
    annotated_conllu_plus: str | None
    annotated_conllu_plus_files: list[str]


def run(
    *,
    data_dir: str | Path = "CobaldRus",
    hierarchy: str | Path | pd.DataFrame | None = "hyperonims_hierarchy.csv",
    splits: Sequence[str] = ("train", "dev"),
    algorithms: Sequence[str] = ("BisectingKMeans", "BIRCH", "KNNLeiden"),
    n_clusters: int | str | Sequence[int] = "data_semclass",
    output_dir: str | Path = DEFAULT_COLAB_DIR,
    results_dir: str | Path | None = None,
    save_embeddings_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    embedding_filename: str = "gigachat3_10b_a1_8b_base_cobald.pkl",
    embeddings_path: str | Path | None = None,
    training_source: str = COBALD_TRAINING_SOURCE,
    training_embedding_filename: str | None = None,
    training_embeddings_path: str | Path | None = None,
    syntagrus_dir: str | Path = DEFAULT_SYNTAGRUS_DIR,
    syntagrus_files: Sequence[str] | None = DEFAULT_SYNTAGRUS_FILES,
    syntagrus_repo_url: str = DEFAULT_SYNTAGRUS_REPO_URL,
    clone_syntagrus_if_missing: bool = False,
    clustering_features_filename: str | None = None,
    training_clustering_features_filename: str | None = None,
    excel_filename: str = "gigachat3_10b_a1_8b_base_cluster_summaries.xlsx",
    hierarchy_alignment_filename: str = (
        "gigachat3_10b_a1_8b_base_hierarchy_alignment.xlsx"
    ),
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
    n_init: int = 1,
    min_samples: int = 10,
    min_cluster_size: int = 25,
    hdbscan_cluster_selection_epsilon: float = 0.0,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_allow_single_cluster: bool = False,
    hdbscan_n_jobs: int | None = -1,
    metric: str = "euclidean",
    birch_threshold: float = 0.75,
    birch_branching_factor: int = 100,
    bisecting_strategy: str = "biggest_inertia",
    graph_n_neighbors: int = 15,
    graph_resolution: float = 1.0,
    graph_metric: str = "cosine",
    graph_n_jobs: int | None = -1,
    hierarchy_depths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
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
    if training_clustering_features_filename is None:
        training_clustering_features_filename = (
            "gigachat3_10b_a1_8b_base_training_"
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
        trust_remote_code=trust_remote_code,
        include_punctuation_context=include_punctuation_context,
    )
    payload, training_payload, resolved_training_source, training_data_paths = (
        prepare_training_and_cobald_payloads(
            data_dir=data_dir,
            splits=splits,
            training_source=training_source,
            output_dir=output_dir,
            embedding_output_path=embedding_output_path,
            training_embedding_filename=training_embedding_filename,
            embeddings_path=embeddings_path,
            training_embeddings_path=training_embeddings_path,
            drive_output_dir=drive_output_dir,
            embedding_config=embedding_config,
            syntagrus_dir=syntagrus_dir,
            syntagrus_files=syntagrus_files,
            syntagrus_repo_url=syntagrus_repo_url,
            clone_syntagrus_if_missing=clone_syntagrus_if_missing,
            show_progress=show_progress,
            generate_cobald_embeddings=generate_embeddings_from_corpus,
        )
    )
    print(f"embeddings_shape={payload['embeddings'].shape}")
    print(f"tokens={len(payload['tokens'])}")
    print(f"embeddings_saved={payload['saved_path']}")
    print(f"training_source={resolved_training_source}")
    print(f"training_embeddings_shape={training_payload['embeddings'].shape}")
    print(f"training_tokens={len(training_payload['tokens'])}")
    print(f"training_embeddings_saved={training_payload['saved_path']}")
    if training_data_paths:
        print(f"training_data_paths={training_data_paths}")

    if embeddings_path is None and release_cuda_after_embeddings:
        _release_cuda_memory()

    if resolved_training_source == COBALD_TRAINING_SOURCE:
        clustering_payload = make_clustering_payload(
            payload,
            projection_n_components=projection_n_components,
            projection_batch_size=projection_batch_size,
            normalize_projection_input=normalize_projection_input,
            normalize_projection_output=normalize_projection_output,
            show_progress=show_progress,
        )
        training_clustering_payload = clustering_payload
        training_clustering_features_path = None
    else:
        training_clustering_payload, clustering_payload = (
            make_training_and_cobald_clustering_payloads(
                training_payload=training_payload,
                cobald_payload=payload,
                projection_n_components=projection_n_components,
                projection_batch_size=projection_batch_size,
                normalize_projection_input=normalize_projection_input,
                normalize_projection_output=normalize_projection_output,
                show_progress=show_progress,
            )
        )
        training_clustering_features_path = save_embeddings_pickle(
            training_clustering_payload,
            output_dir / training_clustering_features_filename,
            drive_dir=drive_output_dir,
        )
        training_clustering_payload["saved_path"] = str(
            training_clustering_features_path
        )
    clustering_features_path = save_embeddings_pickle(
        clustering_payload,
        output_dir / clustering_features_filename,
        drive_dir=drive_output_dir,
    )
    clustering_payload["saved_path"] = str(clustering_features_path)
    print(
        "training_clustering_features_shape="
        f"{training_clustering_payload['embeddings'].shape}"
    )
    if training_clustering_features_path is not None:
        print(f"training_clustering_features_saved={training_clustering_features_path}")
    print(f"clustering_features_shape={clustering_payload['embeddings'].shape}")
    print(f"clustering_features_saved={clustering_features_path}")

    configs = build_cluster_configs(
        algorithms=algorithms,
        n_clusters=n_clusters,
        tokens=clustering_payload["tokens"],
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
    _guard_quadratic_algorithms(
        configs,
        n_rows=len(training_clustering_payload["tokens"]),
        agglomerative_max_rows=agglomerative_max_rows,
        allow_quadratic_algorithms=allow_quadratic_algorithms,
    )

    results: list[dict[str, object]] = []
    progress = tqdm(configs, desc="Clustering full dataset", disable=not show_progress)
    for config in progress:
        progress.set_postfix(algorithm=config.algorithm)
        if resolved_training_source == COBALD_TRAINING_SOURCE:
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
        else:
            results.append(
                run_clustering_with_training_data(
                    training_clustering_payload["embeddings"],
                    clustering_payload["embeddings"],
                    clustering_payload["tokens"],
                    config=config,
                    hierarchy=hierarchy,
                    hierarchy_depths=hierarchy_depths,
                    show_progress=show_progress,
                )
            )

    result_output_dir = (
        Path(results_dir)
        if results_dir is not None
        else default_results_dir(
            output_dir,
            drive_dir,
            default_output_dir=DEFAULT_COLAB_DIR,
        )
    )
    export_paths = write_clustering_outputs(
        results,
        clustering_payload["tokens"],
        output_dir=result_output_dir,
        label=label,
        metadata={
            "model_name": MODEL_NAME,
            "training_source": resolved_training_source,
            "training_data_paths": training_data_paths,
            "projection_n_components": projection_n_components,
            "hierarchy_depths": list(hierarchy_depths),
        },
    )

    paths = GigaChatBaselinePaths(
        embeddings=str(payload["saved_path"]),
        clustering_features=str(clustering_features_path),
        training_embeddings=(
            str(training_payload["saved_path"])
            if resolved_training_source != COBALD_TRAINING_SOURCE
            else None
        ),
        training_clustering_features=(
            str(training_clustering_features_path)
            if training_clustering_features_path is not None
            else None
        ),
        excel=export_paths.excel,
        semclass_cluster_map_excel=export_paths.semclass_cluster_map_excel,
        hierarchy_alignment_excel=export_paths.hierarchy_alignment_excel,
        scores_csv=export_paths.scores_csv,
        scores_xlsx=export_paths.scores_xlsx,
        scores_txt=export_paths.scores_txt,
        artifacts_pickle=export_paths.artifacts_pickle,
        annotated_conllu_plus=export_paths.annotated_conllu_plus,
        annotated_conllu_plus_files=export_paths.annotated_conllu_plus_files,
    )
    return {
        "embedding_payload": payload,
        "training_embedding_payload": training_payload,
        "clustering_payload": clustering_payload,
        "training_clustering_payload": training_clustering_payload,
        "results": results,
        "paths": asdict(paths),
        "training_source": resolved_training_source,
        "training_data_paths": training_data_paths,
    }

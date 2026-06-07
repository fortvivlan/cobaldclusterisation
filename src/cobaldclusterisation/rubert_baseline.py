"""Colab-friendly ruBERT-tiny2 baseline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .baseline_payloads import load_saved_embedding_payload
from .cluster_exports import default_results_dir, write_clustering_outputs
from .clustering import (
    ClusterConfig,
    infer_semclass_cluster_count,
    run_clustering,
    run_clustering_with_training_data,
)
from .data import Sentence, load_corpus, load_ud_corpus
from .embeddings import (
    EmbeddingConfig,
    generate_embeddings_from_corpus,
    generate_token_embeddings,
    save_embeddings_pickle,
)
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
COBALD_TRAINING_SOURCE = "cobald"
SYNTAGRUS_TRAINING_SOURCE = "syntagrus"
MERGED_TRAINING_SOURCE = "syntagrus_cobald"
DEFAULT_SYNTAGRUS_REPO_URL = (
    "https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git"
)
DEFAULT_SYNTAGRUS_DIR = "UD_Russian-SynTagRus"
DEFAULT_SYNTAGRUS_FILES = (
    "ru_syntagrus-ud-train-a.conllu",
    "ru_syntagrus-ud-train-b.conllu",
    "ru_syntagrus-ud-train-c.conllu",
)


@dataclass(slots=True)
class RubertBaselinePaths:
    """Output paths produced by :func:`run`."""

    embeddings: str
    training_embeddings: str | None
    excel: str
    semclass_cluster_map_excel: str
    hierarchy_alignment_excel: str
    scores_csv: str
    scores_xlsx: str
    scores_txt: list[str]
    artifacts_pickle: str
    annotated_conllu_plus: str | None
    annotated_conllu_plus_files: list[str]


def _clone_repo_if_missing(repo_url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(destination)], check=True)


def _normalize_training_source(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "cobald": COBALD_TRAINING_SOURCE,
        "cobaldrus": COBALD_TRAINING_SOURCE,
        "syntagrus": SYNTAGRUS_TRAINING_SOURCE,
        "syntagrus_ud": SYNTAGRUS_TRAINING_SOURCE,
        "ud_syntagrus": SYNTAGRUS_TRAINING_SOURCE,
        "merged": MERGED_TRAINING_SOURCE,
        "merge": MERGED_TRAINING_SOURCE,
        "cobald_syntagrus": MERGED_TRAINING_SOURCE,
        "syntagrus_cobald": MERGED_TRAINING_SOURCE,
        "syntagrus_and_cobald": MERGED_TRAINING_SOURCE,
        "cobald_and_syntagrus": MERGED_TRAINING_SOURCE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "training_source must be one of 'cobald', 'syntagrus', "
            "or 'syntagrus_cobald'."
        ) from exc


def _resolve_syntagrus_paths(
    syntagrus_dir: str | Path,
    *,
    syntagrus_files: Sequence[str] | None,
    clone_syntagrus_if_missing: bool,
    syntagrus_repo_url: str,
) -> list[Path]:
    path = Path(syntagrus_dir)
    if clone_syntagrus_if_missing:
        _clone_repo_if_missing(syntagrus_repo_url, path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(
            f"SynTagRus directory not found: {path}. Clone "
            f"{syntagrus_repo_url} there, pass syntagrus_dir, or set "
            "clone_syntagrus_if_missing=True."
        )
    names = (
        tuple(syntagrus_files)
        if syntagrus_files is not None
        else DEFAULT_SYNTAGRUS_FILES
    )
    paths = [path / name for name in names]
    missing = [candidate for candidate in paths if not candidate.is_file()]
    if missing:
        missing_text = ", ".join(str(candidate) for candidate in missing)
        raise FileNotFoundError(f"Missing SynTagRus CoNLL-U file(s): {missing_text}")
    return paths


def _training_embedding_filename(
    embedding_filename: str,
    *,
    training_source: str,
) -> str:
    path = Path(embedding_filename)
    suffix = path.suffix or ".pkl"
    stem = path.stem if path.suffix else path.name
    return f"{stem}_{training_source}_training{suffix}"


def _save_generated_payload(
    payload: dict[str, object],
    *,
    output_path: Path,
    drive_dir: Path | None,
) -> dict[str, object]:
    saved_path = save_embeddings_pickle(payload, output_path, drive_dir=drive_dir)
    payload["saved_path"] = str(saved_path)
    return payload


def _generate_embeddings_from_sentences(
    sentences: Sequence[Sentence],
    *,
    output_path: Path,
    drive_dir: Path | None,
    config: EmbeddingConfig,
    show_progress: bool,
) -> dict[str, object]:
    payload = generate_token_embeddings(
        sentences,
        config=config,
        show_progress=show_progress,
    )
    return _save_generated_payload(
        payload,
        output_path=output_path,
        drive_dir=drive_dir,
    )


def _merge_embedding_payloads(
    payloads: Sequence[dict[str, object]],
    *,
    source: str,
) -> dict[str, object]:
    embeddings = [
        np.asarray(payload["embeddings"], dtype=np.float32)
        for payload in payloads
    ]
    tokens = [
        payload["tokens"]
        for payload in payloads
        if isinstance(payload.get("tokens"), pd.DataFrame)
    ]
    if len(tokens) != len(payloads):
        raise ValueError("All embedding payloads must contain a token DataFrame")
    return {
        "embeddings": np.vstack(embeddings).astype(np.float32, copy=False),
        "tokens": pd.concat(tokens, ignore_index=True),
        "config": {
            "source": source,
            "parts": [payload.get("saved_path") for payload in payloads],
            "target_policy": "surface_non_punctuation_tokens",
        },
    }


def prepare_training_and_cobald_payloads(
    *,
    data_dir: str | Path,
    splits: Sequence[str],
    training_source: str,
    output_dir: Path,
    embedding_output_path: Path,
    training_embedding_filename: str | None,
    embeddings_path: str | Path | None,
    training_embeddings_path: str | Path | None,
    drive_output_dir: Path | None,
    embedding_config: EmbeddingConfig,
    syntagrus_dir: str | Path,
    syntagrus_files: Sequence[str] | None,
    syntagrus_repo_url: str,
    clone_syntagrus_if_missing: bool,
    show_progress: bool,
    generate_cobald_embeddings: Callable[..., dict[str, object]] = (
        generate_embeddings_from_corpus
    ),
) -> tuple[dict[str, object], dict[str, object], str, list[str]]:
    """Return ``(cobald_payload, training_payload, source, data_paths)``."""

    resolved_training_source = _normalize_training_source(training_source)

    if embeddings_path is None:
        cobald_payload = generate_cobald_embeddings(
            data_dir=data_dir,
            output_path=embedding_output_path,
            drive_dir=drive_output_dir,
            splits=splits,
            config=embedding_config,
            show_progress=show_progress,
        )
    else:
        cobald_payload = load_saved_embedding_payload(embeddings_path)

    if resolved_training_source == COBALD_TRAINING_SOURCE:
        return cobald_payload, cobald_payload, resolved_training_source, []

    training_output_path = output_dir / (
        training_embedding_filename
        or _training_embedding_filename(
            embedding_output_path.name,
            training_source=resolved_training_source,
        )
    )
    if training_embeddings_path is not None:
        training_payload = load_saved_embedding_payload(training_embeddings_path)
        return cobald_payload, training_payload, resolved_training_source, []

    syntagrus_paths = _resolve_syntagrus_paths(
        syntagrus_dir,
        syntagrus_files=syntagrus_files,
        clone_syntagrus_if_missing=clone_syntagrus_if_missing,
        syntagrus_repo_url=syntagrus_repo_url,
    )
    syntagrus_sentences = load_ud_corpus(syntagrus_paths, split="syntagrus")
    syntagrus_output_path = training_output_path
    if resolved_training_source == MERGED_TRAINING_SOURCE:
        syntagrus_output_path = output_dir / _training_embedding_filename(
            embedding_output_path.name,
            training_source=SYNTAGRUS_TRAINING_SOURCE,
        )
    syntagrus_payload = _generate_embeddings_from_sentences(
        syntagrus_sentences,
        output_path=syntagrus_output_path,
        drive_dir=drive_output_dir,
        config=embedding_config,
        show_progress=show_progress,
    )

    if resolved_training_source == MERGED_TRAINING_SOURCE:
        training_payload = _merge_embedding_payloads(
            [syntagrus_payload, cobald_payload],
            source=resolved_training_source,
        )
        training_payload = _save_generated_payload(
            training_payload,
            output_path=training_output_path,
            drive_dir=drive_output_dir,
        )
    else:
        training_payload = syntagrus_payload

    return (
        cobald_payload,
        training_payload,
        resolved_training_source,
        [str(path) for path in syntagrus_paths],
    )


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
    results_dir: str | Path | None = None,
    save_embeddings_to_drive: bool = False,
    drive_dir: str | Path = DEFAULT_DRIVE_DIR,
    embedding_filename: str = "rubert_tiny2_cobald.pkl",
    embeddings_path: str | Path | None = None,
    training_source: str = COBALD_TRAINING_SOURCE,
    training_embedding_filename: str | None = None,
    training_embeddings_path: str | Path | None = None,
    syntagrus_dir: str | Path = DEFAULT_SYNTAGRUS_DIR,
    syntagrus_files: Sequence[str] | None = DEFAULT_SYNTAGRUS_FILES,
    syntagrus_repo_url: str = DEFAULT_SYNTAGRUS_REPO_URL,
    clone_syntagrus_if_missing: bool = False,
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

    embedding_config = EmbeddingConfig(
        model_name=MODEL_NAME,
        batch_size=embedding_batch_size,
        device=device,
        max_length=max_length,
        seed=seed,
        normalize=normalize_embeddings,
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
        if resolved_training_source == COBALD_TRAINING_SOURCE:
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
        else:
            results.append(
                run_clustering_with_training_data(
                    training_payload["embeddings"],
                    payload["embeddings"],
                    payload["tokens"],
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
        payload["tokens"],
        output_dir=result_output_dir,
        label=label,
        metadata={
            "model_name": MODEL_NAME,
            "training_source": resolved_training_source,
            "training_data_paths": training_data_paths,
            "hierarchy_depths": list(hierarchy_depths),
        },
    )

    paths = RubertBaselinePaths(
        embeddings=str(payload["saved_path"]),
        training_embeddings=(
            str(training_payload["saved_path"])
            if resolved_training_source != COBALD_TRAINING_SOURCE
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
        "results": results,
        "paths": asdict(paths),
        "training_source": resolved_training_source,
        "training_data_paths": training_data_paths,
    }

"""Clustering and evaluation helpers for CoBaLD token embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import normalize as l2_normalize
from tqdm.auto import tqdm

from .data import load_semclass_hierarchy
from .embeddings import load_embeddings_pickle
from .resources import resolve_hierarchy_path


@dataclass(slots=True)
class ClusterConfig:
    """Parameters for one clustering run."""

    algorithm: str = "minibatch_kmeans"
    n_clusters: int = 100
    random_state: int = 42
    normalize: bool = True
    batch_size: int = 4096
    n_init: int = 10
    min_samples: int = 10
    min_cluster_size: int = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"
    allow_single_cluster: bool = False
    n_jobs: int | None = None
    linkage: str = "ward"
    metric: str = "euclidean"
    birch_threshold: float = 0.75
    birch_branching_factor: int = 100
    bisecting_strategy: str = "biggest_inertia"
    graph_n_neighbors: int = 15
    graph_resolution: float = 1.0


def _require_hdbscan() -> object:
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "HDBSCAN requires scikit-learn with sklearn.cluster.HDBSCAN. "
            "In Colab, install or upgrade with: pip install -U scikit-learn"
        ) from exc
    return HDBSCAN


def _require_graph_dependencies() -> tuple[object, object, object]:
    try:
        import igraph as ig
        import leidenalg
        import pynndescent
    except ImportError as exc:
        raise ImportError(
            "kNN graph clustering requires optional graph dependencies. "
            'In Colab, install them with: pip install -e ".[graph]"'
        ) from exc
    return ig, leidenalg, pynndescent


def _as_float_matrix(embeddings: np.ndarray, *, normalize: bool) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    if normalize:
        matrix = l2_normalize(matrix)
    return matrix


def _graph_edge_weight(distance: float, *, metric: str) -> float:
    if metric == "cosine":
        return max(0.0, 1.0 - distance)
    return 1.0 / (1.0 + max(0.0, distance))


def _fit_predict_knn_graph(
    matrix: np.ndarray,
    *,
    config: ClusterConfig,
    method: str,
) -> tuple[np.ndarray, object]:
    ig, leidenalg, pynndescent = _require_graph_dependencies()
    if config.graph_n_neighbors < 1:
        raise ValueError("graph_n_neighbors must be positive")

    index = pynndescent.NNDescent(
        matrix,
        n_neighbors=config.graph_n_neighbors + 1,
        metric=config.metric,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    neighbor_indices, neighbor_distances = index.neighbor_graph

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for source, (targets, distances) in enumerate(
        zip(neighbor_indices, neighbor_distances, strict=True)
    ):
        for target, distance in zip(targets, distances, strict=True):
            target = int(target)
            if source == target:
                continue
            edges.append((source, target))
            weights.append(_graph_edge_weight(float(distance), metric=config.metric))

    graph = ig.Graph(n=len(matrix), edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "max"})
    if method == "knn_leiden":
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=config.graph_resolution,
            seed=config.random_state,
        )
        labels = np.asarray(partition.membership, dtype=np.int64)
    elif method == "knn_louvain":
        partition = graph.community_multilevel(weights="weight")
        labels = np.asarray(partition.membership, dtype=np.int64)
    else:
        raise ValueError(f"Unknown graph community method: {method}")

    model = {
        "algorithm": method,
        "n_neighbors": config.graph_n_neighbors,
        "metric": config.metric,
        "resolution": config.graph_resolution,
        "n_vertices": graph.vcount(),
        "n_edges": graph.ecount(),
        "n_communities": int(len(np.unique(labels))),
    }
    return labels, model


def fit_predict_clusters(
    embeddings: np.ndarray,
    *,
    config: ClusterConfig | None = None,
) -> tuple[np.ndarray, object]:
    """Fit one clustering model and return integer labels plus the model."""

    config = config or ClusterConfig()
    matrix = _as_float_matrix(embeddings, normalize=config.normalize)
    algorithm = config.algorithm.lower()

    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            n_init=config.n_init,
        )
    elif algorithm == "minibatch_kmeans":
        model = MiniBatchKMeans(
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            batch_size=config.batch_size,
            n_init=config.n_init,
        )
    elif algorithm == "bisecting_kmeans":
        model = BisectingKMeans(
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            n_init=config.n_init,
            bisecting_strategy=config.bisecting_strategy,
        )
    elif algorithm == "birch":
        model = Birch(
            threshold=config.birch_threshold,
            branching_factor=config.birch_branching_factor,
            n_clusters=config.n_clusters,
            compute_labels=True,
        )
    elif algorithm == "agglomerative":
        kwargs: dict[str, object] = {
            "n_clusters": config.n_clusters,
            "linkage": config.linkage,
        }
        if config.linkage != "ward":
            kwargs["metric"] = config.metric
        model = AgglomerativeClustering(**kwargs)
    elif algorithm == "hdbscan":
        HDBSCAN = _require_hdbscan()
        model = HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            cluster_selection_method=config.cluster_selection_method,
            allow_single_cluster=config.allow_single_cluster,
            n_jobs=config.n_jobs,
            metric=config.metric,
        )
    elif algorithm in {"knn_leiden", "knn_louvain"}:
        return _fit_predict_knn_graph(matrix, config=config, method=algorithm)
    else:
        raise ValueError(
            "Unknown algorithm. Use one of: kmeans, minibatch_kmeans, "
            "bisecting_kmeans, birch, agglomerative, hdbscan, "
            "knn_leiden, knn_louvain."
        )

    labels = model.fit_predict(matrix)
    return np.asarray(labels, dtype=np.int64), model


def default_cluster_configs(
    *,
    n_clusters: Sequence[int] = (565,),
    random_state: int = 42,
) -> list[ClusterConfig]:
    """Return scalable default clustering configurations for the full corpus."""

    configs: list[ClusterConfig] = []
    for k in n_clusters:
        configs.append(
            ClusterConfig(
                algorithm="bisecting_kmeans",
                n_clusters=k,
                random_state=random_state,
                n_init=1,
            )
        )
        configs.append(
            ClusterConfig(
                algorithm="birch",
                n_clusters=k,
                random_state=random_state,
            )
        )
    configs.append(
        ClusterConfig(
            algorithm="knn_leiden",
            random_state=random_state,
            metric="cosine",
        )
    )
    return configs


def purity_score(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    """Calculate cluster purity against gold labels."""

    if len(y_true) == 0:
        return float("nan")
    table = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="gold"))
    if table.empty:
        return float("nan")
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def _valid_semclass_mask(values: Sequence[object]) -> np.ndarray:
    series = pd.Series(values)
    return series.notna().to_numpy() & ~series.astype(str).isin(
        ["", "_", "nan", "None"]
    ).to_numpy()


def _external_scores(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    prefix: str,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    mask = _valid_semclass_mask(y_true_array)
    if mask.sum() == 0:
        return {
            f"{prefix}_adjusted_rand": float("nan"),
            f"{prefix}_normalized_mutual_info": float("nan"),
            f"{prefix}_homogeneity": float("nan"),
            f"{prefix}_completeness": float("nan"),
            f"{prefix}_v_measure": float("nan"),
            f"{prefix}_purity": float("nan"),
            f"{prefix}_n_labeled": 0.0,
        }
    true_filtered = y_true_array[mask]
    pred_filtered = y_pred_array[mask]
    return {
        f"{prefix}_adjusted_rand": float(adjusted_rand_score(true_filtered, pred_filtered)),
        f"{prefix}_normalized_mutual_info": float(
            normalized_mutual_info_score(true_filtered, pred_filtered)
        ),
        f"{prefix}_homogeneity": float(homogeneity_score(true_filtered, pred_filtered)),
        f"{prefix}_completeness": float(completeness_score(true_filtered, pred_filtered)),
        f"{prefix}_v_measure": float(v_measure_score(true_filtered, pred_filtered)),
        f"{prefix}_purity": purity_score(true_filtered, pred_filtered),
        f"{prefix}_n_labeled": float(mask.sum()),
    }


def _internal_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    max_silhouette_samples: int = 10_000,
    random_state: int = 42,
) -> dict[str, float]:
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    if len(unique_labels) < 2 or mask.sum() <= len(unique_labels):
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "n_clusters_found": float(len(unique_labels)),
            "n_noise": float((labels == -1).sum()),
        }

    matrix = embeddings[mask]
    filtered_labels = labels[mask]
    sample_size = min(max_silhouette_samples, len(filtered_labels))
    return {
        "silhouette": float(
            silhouette_score(
                matrix,
                filtered_labels,
                sample_size=sample_size,
                random_state=random_state,
            )
        ),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, filtered_labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, filtered_labels)),
        "n_clusters_found": float(len(unique_labels)),
        "n_noise": float((labels == -1).sum()),
    }


def hierarchy_ancestor_labels(
    semclasses: Sequence[object],
    hierarchy: pd.DataFrame,
    *,
    depth: int,
) -> np.ndarray:
    """Map SEMCLASS names to ancestor class names at a requested hierarchy depth."""

    required = {"class_id", "parent_id", "depth", "class_name"}
    missing = required.difference(hierarchy.columns)
    if missing:
        raise ValueError(f"Hierarchy DataFrame is missing columns: {sorted(missing)}")

    by_id: dict[str, Mapping[str, object]] = {
        str(row["class_id"]): row
        for row in hierarchy.to_dict(orient="records")
    }
    name_to_id = {
        str(row["class_name"]): str(row["class_id"])
        for row in hierarchy.to_dict(orient="records")
    }

    def ancestor_name(label: object) -> str:
        label_text = str(label)
        if label_text in {"", "_", "nan", "None"}:
            return "_"
        current_id = name_to_id.get(label_text)
        if current_id is None:
            return label_text
        while current_id in by_id and int(by_id[current_id]["depth"]) > depth:
            parent_id = by_id[current_id]["parent_id"]
            if parent_id is None:
                break
            current_id = str(parent_id)
        row = by_id.get(current_id)
        if row is None:
            return label_text
        return str(row["class_name"])

    return np.asarray([ancestor_name(label) for label in semclasses], dtype=object)


def _load_hierarchy_frame(
    hierarchy: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    return (
        load_semclass_hierarchy(resolve_hierarchy_path(hierarchy))
        if isinstance(hierarchy, (str, Path))
        else hierarchy
    )


def infer_semclass_cluster_count(tokens: pd.DataFrame) -> int:
    """Infer a leaf-level cluster count from SEMCLASS labels present in data."""

    if "SEMCLASS" not in tokens.columns:
        raise ValueError("Cannot infer clusters: tokens has no SEMCLASS column")
    semclasses = tokens["SEMCLASS"].astype(str)
    valid = semclasses[~semclasses.isin(["", "_", "nan"])]
    n_classes = int(valid.nunique())
    if n_classes < 1:
        raise ValueError("Cannot infer clusters: no usable SEMCLASS labels found")
    return n_classes


def hierarchy_alignment_table(
    labels: Sequence[int],
    tokens: pd.DataFrame,
    hierarchy: pd.DataFrame | str | Path,
    *,
    hierarchy_depths: Sequence[int] = (1, 2, 3, 4, 5, 6, 7),
) -> pd.DataFrame:
    """Summarize how each cluster aligns to hierarchy labels at each depth."""

    labels_array = np.asarray(labels, dtype=np.int64)
    if len(labels_array) != len(tokens):
        raise ValueError("labels and tokens must have the same length")
    if "SEMCLASS" not in tokens.columns:
        raise ValueError("tokens must contain a SEMCLASS column")

    hierarchy_df = _load_hierarchy_frame(hierarchy)
    semclasses = tokens["SEMCLASS"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    cluster_labels = sorted(
        np.unique(labels_array),
        key=lambda label: (label == -1, int(label)),
    )

    for depth in hierarchy_depths:
        depth_labels = hierarchy_ancestor_labels(semclasses, hierarchy_df, depth=depth)
        valid_mask = _valid_semclass_mask(depth_labels)
        depth_counts = pd.Series(depth_labels[valid_mask]).value_counts()
        for cluster_label in cluster_labels:
            cluster_mask = labels_array == cluster_label
            cluster_valid_mask = cluster_mask & valid_mask
            cluster_size = int(cluster_mask.sum())
            labeled_size = int(cluster_valid_mask.sum())
            if labeled_size == 0:
                rows.append(
                    {
                        "depth": int(depth),
                        "cluster": int(cluster_label),
                        "cluster_token_count": cluster_size,
                        "cluster_labeled_count": 0,
                        "best_label": "",
                        "best_label_count": 0,
                        "cluster_purity": float("nan"),
                        "gold_label_coverage": float("nan"),
                    }
                )
                continue

            cluster_counts = pd.Series(depth_labels[cluster_valid_mask]).value_counts()
            best_label = str(cluster_counts.index[0])
            best_count = int(cluster_counts.iloc[0])
            gold_count = int(depth_counts.get(best_label, 0))
            rows.append(
                {
                    "depth": int(depth),
                    "cluster": int(cluster_label),
                    "cluster_token_count": cluster_size,
                    "cluster_labeled_count": labeled_size,
                    "best_label": best_label,
                    "best_label_count": best_count,
                    "cluster_purity": best_count / labeled_size,
                    "gold_label_coverage": (
                        best_count / gold_count if gold_count else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_clusters(
    embeddings: np.ndarray,
    labels: Sequence[int],
    *,
    tokens: pd.DataFrame | None = None,
    hierarchy: pd.DataFrame | str | Path | None = None,
    hierarchy_depths: Sequence[int] = (1, 2, 3),
    max_silhouette_samples: int = 10_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Calculate internal and SEMCLASS-aware clustering scores."""

    labels_array = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if len(labels_array) != len(matrix):
        raise ValueError(
            f"labels length ({len(labels_array)}) does not match embeddings "
            f"length ({len(matrix)})"
        )

    scores = _internal_scores(
        matrix,
        labels_array,
        max_silhouette_samples=max_silhouette_samples,
        random_state=random_state,
    )
    if tokens is None or "SEMCLASS" not in tokens.columns:
        return scores

    semclasses = tokens["SEMCLASS"].astype(str).to_numpy()
    scores.update(_external_scores(semclasses, labels_array, prefix="semclass_exact"))

    if hierarchy is not None:
        hierarchy_df = _load_hierarchy_frame(hierarchy)
        for depth in hierarchy_depths:
            ancestor_labels = hierarchy_ancestor_labels(
                semclasses,
                hierarchy_df,
                depth=depth,
            )
            scores.update(
                _external_scores(
                    ancestor_labels,
                    labels_array,
                    prefix=f"semclass_depth_{depth}",
                )
            )
    return scores


def _format_value_counts(series: pd.Series, *, top_n: int | None = None) -> str:
    values = series.astype(str)
    values = values[~values.isin(["", "_", "nan"])]
    counts = values.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    return "\n".join(f"{value}: {count}" for value, count in counts.items())


def _format_semclass_lemma_examples(
    tokens: pd.DataFrame,
    *,
    rng: np.random.Generator,
    max_lemmas_per_semclass: int = 10,
) -> str:
    semclasses = tokens["SEMCLASS"].astype(str)
    valid_semclasses = semclasses[~semclasses.isin(["", "_", "nan"])]
    semclass_counts = valid_semclasses.value_counts()
    lines: list[str] = []
    for semclass in semclass_counts.index:
        lemmas = (
            tokens.loc[semclasses == semclass, "LEMMA"]
            .astype(str)
            .loc[lambda values: ~values.isin(["", "_", "nan"])]
            .drop_duplicates()
            .to_numpy(dtype=object)
        )
        if len(lemmas) == 0:
            continue
        if len(lemmas) > max_lemmas_per_semclass:
            lemmas = rng.choice(lemmas, size=max_lemmas_per_semclass, replace=False)
        lines.append(f"{semclass}: {', '.join(str(lemma) for lemma in lemmas)}")
    return "\n".join(lines)


def summarize_clusters(
    embeddings: np.ndarray,
    labels: Sequence[int],
    tokens: pd.DataFrame,
    *,
    examples_per_cluster: int = 5,
    top_n: int = 8,
    semclass_lemma_examples: int = 10,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Create human-readable cluster names, examples, and label summaries."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(matrix) != len(tokens) or len(labels_array) != len(tokens):
        raise ValueError("embeddings, labels, and tokens must have the same length")

    token_table = tokens.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []
    cluster_labels = sorted(
        np.unique(labels_array),
        key=lambda label: (label == -1, int(label)),
    )
    for cluster_label in cluster_labels:
        indices = np.flatnonzero(labels_array == cluster_label)
        cluster_vectors = matrix[indices]
        centroid = cluster_vectors.mean(axis=0)
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        representative_index = int(indices[int(np.argmin(distances))])
        representative = token_table.iloc[representative_index]
        cluster_tokens = token_table.iloc[indices]

        representative_lemma = str(representative.get("LEMMA", ""))
        representative_form = str(representative.get("FORM", ""))
        cluster_name = representative_lemma if representative_lemma != "_" else representative_form
        if cluster_label == -1:
            cluster_name = f"NOISE:{cluster_name}"

        examples = []
        for _, example in cluster_tokens.head(examples_per_cluster).iterrows():
            examples.append(
                f"{example.get('FORM', '')} "
                f"[{example.get('SEMCLASS', '_')}] :: {example.get('context_text', '')}"
            )

        rows.append(
            {
                "cluster": int(cluster_label),
                "cluster_name": cluster_name,
                "token_count": int(len(indices)),
                "lemma_count": int(
                    cluster_tokens["LEMMA"]
                    .astype(str)
                    .loc[lambda values: ~values.isin(["", "_", "nan"])]
                    .nunique()
                ),
                "lemmas": _format_value_counts(cluster_tokens["LEMMA"]),
                "top_forms": _format_value_counts(
                    cluster_tokens["FORM"],
                    top_n=top_n,
                ),
                "semclasses": _format_value_counts(cluster_tokens["SEMCLASS"]),
                "semclass_lemma_examples": _format_semclass_lemma_examples(
                    cluster_tokens,
                    rng=rng,
                    max_lemmas_per_semclass=semclass_lemma_examples,
                ),
                "examples": "\n".join(examples),
            }
        )
    return pd.DataFrame(rows)


def run_clustering(
    embeddings: np.ndarray,
    tokens: pd.DataFrame,
    *,
    config: ClusterConfig | None = None,
    hierarchy: pd.DataFrame | str | Path | None = None,
    hierarchy_depths: Sequence[int] = (1, 2, 3),
    show_progress: bool = False,
) -> dict[str, object]:
    """Run one clustering configuration and return labels, scores, and summaries."""

    config = config or ClusterConfig()
    matrix = _as_float_matrix(embeddings, normalize=config.normalize)
    run_label = (
        f"{config.algorithm} min_cluster_size={config.min_cluster_size}"
        if config.algorithm.lower() == "hdbscan"
        else f"{config.algorithm} k={config.n_clusters}"
    )
    progress = tqdm(
        total=3,
        desc=run_label,
        disable=not show_progress,
        leave=False,
    )

    try:
        labels, model = fit_predict_clusters(embeddings, config=config)
        progress.update(1)
        scores = evaluate_clusters(
            matrix,
            labels,
            tokens=tokens,
            hierarchy=hierarchy,
            hierarchy_depths=hierarchy_depths,
            random_state=config.random_state,
        )
        progress.update(1)
        summary = summarize_clusters(
            matrix,
            labels,
            tokens,
            random_state=config.random_state,
        )
        progress.update(1)
        hierarchy_alignment = (
            hierarchy_alignment_table(
                labels,
                tokens,
                hierarchy,
                hierarchy_depths=hierarchy_depths,
            )
            if hierarchy is not None and "SEMCLASS" in tokens.columns
            else pd.DataFrame()
        )
    finally:
        progress.close()
    return {
        "labels": labels,
        "scores": scores,
        "summary": summary,
        "hierarchy_alignment": hierarchy_alignment,
        "model": model,
        "config": asdict(config),
    }


def run_clustering_suite(
    embedding_payload: dict[str, object] | str | Path,
    *,
    configs: Sequence[ClusterConfig] | None = None,
    hierarchy: pd.DataFrame | str | Path | None = None,
    hierarchy_depths: Sequence[int] = (1, 2, 3),
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Run several clustering configurations against an embedding pickle payload."""

    payload = (
        load_embeddings_pickle(embedding_payload)
        if isinstance(embedding_payload, (str, Path))
        else embedding_payload
    )
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    tokens = payload["tokens"]
    configs = list(configs or default_cluster_configs())
    results: list[dict[str, object]] = []
    progress = tqdm(configs, desc="Clustering runs", disable=not show_progress)
    for config in progress:
        algorithm = config.algorithm.lower()
        progress.set_postfix(
            algorithm=config.algorithm,
            k=config.n_clusters if algorithm != "hdbscan" else "hdbscan",
        )
        results.append(
            run_clustering(
                embeddings,
                tokens,
                config=config,
                hierarchy=hierarchy,
                hierarchy_depths=hierarchy_depths,
                show_progress=False,
            )
        )
    return results


def save_clustering_results(
    results: Sequence[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Save clustering results to a pickle file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(list(results), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def scores_to_dataframe(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Convert a clustering-suite result list to a compact scores table."""

    rows: list[dict[str, object]] = []
    for result in results:
        config = result.get("config", {})
        scores = result.get("scores", {})
        rows.append({**config, **scores})
    return pd.DataFrame(rows)

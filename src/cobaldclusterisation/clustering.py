"""Clustering and evaluation helpers for CoBaLD token embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
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
    eps: float = 0.5
    min_samples: int = 10
    linkage: str = "ward"
    metric: str = "euclidean"


def _as_float_matrix(embeddings: np.ndarray, *, normalize: bool) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    if normalize:
        matrix = l2_normalize(matrix)
    return matrix


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
    elif algorithm == "agglomerative":
        kwargs: dict[str, object] = {
            "n_clusters": config.n_clusters,
            "linkage": config.linkage,
        }
        if config.linkage != "ward":
            kwargs["metric"] = config.metric
        model = AgglomerativeClustering(**kwargs)
    elif algorithm == "dbscan":
        model = DBSCAN(
            eps=config.eps,
            min_samples=config.min_samples,
            metric=config.metric,
        )
    else:
        raise ValueError(
            "Unknown algorithm. Use one of: kmeans, minibatch_kmeans, "
            "agglomerative, dbscan."
        )

    labels = model.fit_predict(matrix)
    return np.asarray(labels, dtype=np.int64), model


def default_cluster_configs(
    *,
    n_clusters: Sequence[int] = (50, 100, 200),
    random_state: int = 42,
) -> list[ClusterConfig]:
    """Return scalable default clustering configurations for the full corpus."""

    configs: list[ClusterConfig] = []
    for k in n_clusters:
        configs.append(
            ClusterConfig(
                algorithm="minibatch_kmeans",
                n_clusters=k,
                random_state=random_state,
            )
        )
        configs.append(
            ClusterConfig(
                algorithm="kmeans",
                n_clusters=k,
                random_state=random_state,
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
    return series.notna().to_numpy() & ~series.astype(str).isin(["", "_"]).to_numpy()


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
        hierarchy_df = (
            load_semclass_hierarchy(resolve_hierarchy_path(hierarchy))
            if isinstance(hierarchy, (str, Path))
            else hierarchy
        )
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


def _format_top_values(series: pd.Series, *, top_n: int) -> str:
    values = series.astype(str)
    values = values[~values.isin(["", "_", "nan"])]
    counts = values.value_counts().head(top_n)
    return "; ".join(f"{value}:{count}" for value, count in counts.items())


def summarize_clusters(
    embeddings: np.ndarray,
    labels: Sequence[int],
    tokens: pd.DataFrame,
    *,
    examples_per_cluster: int = 5,
    top_n: int = 8,
) -> pd.DataFrame:
    """Create human-readable cluster names, examples, and label summaries."""

    matrix = np.asarray(embeddings, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(matrix) != len(tokens) or len(labels_array) != len(tokens):
        raise ValueError("embeddings, labels, and tokens must have the same length")

    token_table = tokens.reset_index(drop=True)
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
                "size": int(len(indices)),
                "representative_form": representative_form,
                "representative_lemma": representative_lemma,
                "representative_semclass": str(representative.get("SEMCLASS", "_")),
                "top_lemmas": _format_top_values(cluster_tokens["LEMMA"], top_n=top_n),
                "top_forms": _format_top_values(cluster_tokens["FORM"], top_n=top_n),
                "top_semclasses": _format_top_values(
                    cluster_tokens["SEMCLASS"],
                    top_n=top_n,
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
) -> dict[str, object]:
    """Run one clustering configuration and return labels, scores, and summaries."""

    config = config or ClusterConfig()
    matrix = _as_float_matrix(embeddings, normalize=config.normalize)
    labels, model = fit_predict_clusters(embeddings, config=config)
    scores = evaluate_clusters(
        matrix,
        labels,
        tokens=tokens,
        hierarchy=hierarchy,
        hierarchy_depths=hierarchy_depths,
        random_state=config.random_state,
    )
    summary = summarize_clusters(matrix, labels, tokens)
    return {
        "labels": labels,
        "scores": scores,
        "summary": summary,
        "model": model,
        "config": asdict(config),
    }


def run_clustering_suite(
    embedding_payload: dict[str, object] | str | Path,
    *,
    configs: Sequence[ClusterConfig] | None = None,
    hierarchy: pd.DataFrame | str | Path | None = None,
    hierarchy_depths: Sequence[int] = (1, 2, 3),
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
    return [
        run_clustering(
            embeddings,
            tokens,
            config=config,
            hierarchy=hierarchy,
            hierarchy_depths=hierarchy_depths,
        )
        for config in configs
    ]


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

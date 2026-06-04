from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobaldclusterisation.clustering import ClusterConfig
from cobaldclusterisation import sambalingo_baseline
from cobaldclusterisation.sambalingo_baseline import (
    _guard_quadratic_algorithms,
    make_clustering_payload,
    reduce_embeddings_incremental_pca,
)


def _tokens(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FORM": [f"токен{i}" for i in range(n_rows)],
            "LEMMA": [f"лемма{i}" for i in range(n_rows)],
            "SEMCLASS": ["CLASS"] * n_rows,
            "context_text": [f"контекст {i}" for i in range(n_rows)],
        }
    )


def test_reduce_embeddings_incremental_pca_uses_all_rows() -> None:
    embeddings = np.arange(42, dtype=np.float32).reshape(7, 6)

    reduced = reduce_embeddings_incremental_pca(
        embeddings,
        n_components=2,
        batch_size=3,
        show_progress=False,
    )

    assert reduced.shape == (7, 2)
    assert reduced.dtype == np.float32
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0)


def test_make_clustering_payload_records_projection_config() -> None:
    payload = {
        "embeddings": np.arange(48, dtype=np.float32).reshape(8, 6),
        "tokens": _tokens(8),
        "saved_path": "/content/full.pkl",
        "config": {"model_name": "model"},
    }

    reduced_payload = make_clustering_payload(
        payload,
        projection_n_components=2,
        projection_batch_size=4,
        show_progress=False,
    )

    assert reduced_payload["embeddings"].shape == (8, 2)
    assert len(reduced_payload["tokens"]) == 8
    assert reduced_payload["config"]["source_embedding_path"] == "/content/full.pkl"
    assert reduced_payload["config"]["projection_n_components"] == 2


def test_agglomerative_guard_requires_explicit_override() -> None:
    configs = [ClusterConfig(algorithm="agglomerative", n_clusters=2)]

    with pytest.raises(ValueError, match="allow_quadratic_algorithms=True"):
        _guard_quadratic_algorithms(
            configs,
            n_rows=10,
            agglomerative_max_rows=5,
            allow_quadratic_algorithms=False,
        )

    _guard_quadratic_algorithms(
        configs,
        n_rows=10,
        agglomerative_max_rows=5,
        allow_quadratic_algorithms=True,
    )


def test_sambalingo_run_clusters_complete_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokens = _tokens(8)
    embeddings = np.arange(48, dtype=np.float32).reshape(8, 6)
    seen_rows: list[int] = []

    def fake_generate_embeddings_from_corpus(**kwargs: object) -> dict[str, object]:
        return {
            "embeddings": embeddings,
            "tokens": tokens,
            "saved_path": str(tmp_path / "full.pkl"),
            "config": {"model_name": "sambalingo"},
        }

    def fake_run_clustering(
        clustering_embeddings: np.ndarray,
        clustering_tokens: pd.DataFrame,
        **kwargs: object,
    ) -> dict[str, object]:
        seen_rows.append(len(clustering_tokens))
        assert clustering_embeddings.shape == (8, 2)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        return {
            "labels": labels,
            "scores": {"silhouette": 0.5, "n_noise": 0},
            "summary": pd.DataFrame(
                {
                    "cluster": [0, 1],
                    "cluster_name": ["a", "b"],
                    "token_count": [4, 4],
                }
            ),
            "model": object(),
            "config": {"algorithm": "kmeans", "n_clusters": 2, "normalize": False},
        }

    def fake_write_cluster_summary_excel(
        results: list[dict[str, object]],
        output_path: Path,
    ) -> str:
        output_path.write_text("excel", encoding="utf-8")
        return str(output_path)

    def fake_write_scores(
        results: list[dict[str, object]],
        *,
        output_dir: Path,
        label: str,
    ) -> tuple[str, str, list[str]]:
        scores_csv = output_dir / f"{label}_scores.csv"
        scores_xlsx = output_dir / f"{label}_scores.xlsx"
        scores_txt = output_dir / f"{label}_scores.txt"
        scores_csv.write_text("metric,run_1\nsilhouette,0.5\n", encoding="utf-8")
        scores_xlsx.write_text("excel", encoding="utf-8")
        scores_txt.write_text("scores", encoding="utf-8")
        return str(scores_csv), str(scores_xlsx), [str(scores_txt)]

    monkeypatch.setattr(
        sambalingo_baseline,
        "generate_embeddings_from_corpus",
        fake_generate_embeddings_from_corpus,
    )
    monkeypatch.setattr(sambalingo_baseline, "run_clustering", fake_run_clustering)
    monkeypatch.setattr(
        sambalingo_baseline,
        "write_cluster_summary_excel",
        fake_write_cluster_summary_excel,
    )
    monkeypatch.setattr(sambalingo_baseline, "_write_scores", fake_write_scores)

    result = sambalingo_baseline.run(
        algorithms=["KMeans"],
        n_clusters=2,
        output_dir=tmp_path,
        projection_n_components=2,
        projection_batch_size=4,
        show_progress=False,
    )

    assert seen_rows == [8]
    assert Path(result["paths"]["clustering_features"]).exists()
    assert Path(result["paths"]["excel"]).exists()
    assert Path(result["paths"]["scores_csv"]).exists()
    assert Path(result["paths"]["scores_xlsx"]).exists()
    assert all(Path(path).exists() for path in result["paths"]["scores_txt"])
